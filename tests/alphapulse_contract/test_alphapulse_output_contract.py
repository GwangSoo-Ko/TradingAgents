"""alpha-pulse output contract: the exit code, stdout grammar and report tree of main.py.

alpha-pulse runs ``python <checkout>/main.py TICKER [DATE]`` as a subprocess and reads only:

* the exit code — rc != 0 marks the run failed and discards the plan it already printed;
* the LAST ``Report saved: <path>`` line (its parent directory is the run's report
  directory) and the last bare rating line before it — the decision it shows on its
  dashboard, in its notifications and in the discovery deep report;
* the LAST ``TRADE_PLAN_JSON: {...}`` line (order drafts; no line = no plan, stored as
  plan_status 'unparsed');
* ``<report dir>/complete_report.md`` — its web report view and its discovery input.

Every test here runs the unmodified main.py in a subprocess through the harness (real
graph with ``debug=True``, real nodes, real report writer, real signal path; only the LLM
and the data libraries are faked) and reads the result with ``contract_reader``, a
clean-room reader of that grammar written from the fork's own documentation
(docs/INTEGRATION.md §1b, main.py's comments). Expected values are literals written here
or the scenario's scripted model output — never recomputed by fork code. Each docstring
names the production change the test catches.
"""

from __future__ import annotations

import datetime
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest

from . import contract_reader, scenarios
from ._compat import canonical_json, project_to_expected
from .harness import (
    DECISION_WORDS,
    PACKAGE_DIR,
    REPO_ROOT,
    REPORT_SAVED_LINE_RE,
    RunResult,
    base_env,
    harness_python,
    kst_today,
    run_main,
)

S1 = "s1_nightly_kr_holding_sell"
S2 = "s2_pm_prose_quotes_other_ratings"
S2U = "s2u_pm_prose_quotes_other_ratings_underweight"
S3 = "s3_pm_structured_failure_fallback"
S4 = "s4_us_not_held_discovery"
S5A = "s5a_nightly_yyyymmdd_not_held"
S5B = "s5b_no_date_kst_today"

# The trade date every dated scenario must reach propagate as (argv '20260819' or '2026-08-19').
TRADE_DATE = "2026-08-19"

# Runs whose Portfolio Manager returned a typed plan:
# scenario -> (typed rating, ticker as argv and report-dir component, report title label).
STRUCTURED_RUNS = {
    S1: ("Sell", "417310.KS", "하나글로벌리츠 (417310.KS)"),
    S4: ("Overweight", "AAPL", "Apple Inc. (AAPL)"),
    S5A: ("Buy", "005930.KS", "삼성전자 (005930.KS)"),
}

# main.py drops these two prose fields from the TRADE_PLAN_JSON line (they are in the report).
PLAN_PROSE_FIELDS = ("executive_summary", "investment_thesis")

# Files every completed run must leave in its report directory (more are allowed).
REQUIRED_REPORT_FILES = [
    "1_analysts/fundamentals.md",
    "1_analysts/market.md",
    "1_analysts/news.md",
    "1_analysts/sentiment.md",
    "2_research/bear.md",
    "2_research/bull.md",
    "2_research/manager.md",
    "3_trading/trader.md",
    "4_risk/aggressive.md",
    "4_risk/conservative.md",
    "4_risk/neutral.md",
    "5_portfolio/decision.md",
    "complete_report.md",
]

# The writer's own headings, in the order complete_report.md must carry them.
WRITER_HEADINGS = [
    "## I. Analyst Team Reports",
    "### Market Analyst",
    "### Sentiment Analyst",
    "### News Analyst",
    "### Fundamentals Analyst",
    "## II. Research Team Decision",
    "### Bull Researcher",
    "### Bear Researcher",
    "### Research Manager",
    "## III. Trading Team Plan",
    "### Trader",
    "## IV. Risk Management Team Decision",
    "### Aggressive Analyst",
    "### Conservative Analyst",
    "### Neutral Analyst",
    "## V. Portfolio Manager Decision",
    "### Portfolio Manager",
]

# The per-role models main.build_config() runs, as the report header must list them.
ROLE_MODELS = {
    "market_analyst": ("vertex_anthropic", "claude-sonnet-5"),
    "sentiment_analyst": ("vertex_anthropic", "claude-sonnet-5"),
    "news_analyst": ("vertex_anthropic", "claude-sonnet-5"),
    "fundamentals_analyst": ("vertex_anthropic", "claude-sonnet-5"),
    "bull_researcher": ("vertex_anthropic", "claude-sonnet-5"),
    "bear_researcher": ("vertex_anthropic", "claude-sonnet-5"),
    "research_manager": ("vertex_anthropic", "claude-opus-5"),
    "trader": ("vertex_anthropic", "claude-sonnet-5"),
    "aggressive_debator": ("vertex_anthropic", "claude-sonnet-5"),
    "conservative_debator": ("vertex_anthropic", "claude-sonnet-5"),
    "neutral_debator": ("vertex_anthropic", "claude-sonnet-5"),
    "portfolio_manager": ("vertex_anthropic", "claude-opus-5"),
}
_MODEL_ROW_RE = re.compile(
    r"^\| (?P<role>[a-z_]+)(?: \*\(tier default\)\*)? "
    r"\| `(?P<provider>[^`]*)` \| `(?P<model>[^`]*)` \|$"
)
_MEMORY_TAG_RE = re.compile(
    r"^\[(?P<date>\d{4}-\d{2}-\d{2}) \| (?P<ticker>[^|\]]+?) \| (?P<rating>[^|\]]+?) \|.*\]$"
)

# One quote of another rating per case, each written where a Portfolio Manager writes it
# in production prose. Every case alone flips a "last 'Rating:' label wins" parser away
# from the typed Sell. Value: (patch of the PM's raw tool args, text that must reach the
# rendered decision so the case is armed).
_KILL_SWITCH_CONDITION = scenarios.sell_plan()["kill_switch"]["condition"]
QUOTE_CASES: dict[str, tuple[dict[str, Any], str]] = {
    "executive_summary_quotes_prior_buy": (
        {"executive_summary": scenarios.REIT_SELL_EXEC_SUMMARY + " Most recent prior rating: Buy."},
        "Most recent prior rating: Buy.",
    ),
    "revision_note_quotes_founding_overweight": (
        {"revision": {"kind": "thesis_error",
                      "note": "The founding rating - Overweight - assumed a dividend capacity "
                              "that the higher refinancing rate no longer supports."}},
        "The founding rating - Overweight -",
    ),
    "kill_switch_quotes_consensus_hold": (
        {"kill_switch": {"condition": _KILL_SWITCH_CONDITION + " (consensus rating: Hold)"}},
        "(consensus rating: Hold)",
    ),
    "korean_thesis_quotes_entry_rating_buy": (
        {"investment_thesis": "진입 당시 등급(Rating: Buy)의 전제였던 배당 여력이 차환 금리 상승으로 "
                              "줄었다. " + scenarios.REIT_SELL_THESIS},
        "진입 당시 등급(Rating: Buy)",
    ),
}
# case -> typed rating: the four single quotes (typed Sell) plus the ready scenarios that
# carry all four quotes at once (typed Sell / typed Underweight).
TYPED_UNDER_QUOTES = {**dict.fromkeys(QUOTE_CASES, "Sell"), S2: "Sell", S2U: "Underweight"}

# The ways the Portfolio Manager's structured call fails in production. Each must end in
# the fork's free-text fallback: no plan line. Value: patch of s3's PM role (None = s3).
FALLBACK_TRIGGERS: dict[str, dict[str, Any] | None] = {
    "model_answers_in_prose": None,
    "provider_error": {"structured": "raise"},
    "unreadable_stop_loss_number": {"structured": {**scenarios.reit_sell_pm(),
                                                   "stop_loss": "10,380원"}},
}

# A Portfolio Manager fallback that names no rating at all (the model defers the call): no
# Buy/Overweight/Hold/Underweight/Sell anywhere, no 'Rating:' label. s3's PM_FREE_TEXT names
# three ratings, so it never reaches the "no rating" path.
PM_FREE_TEXT_WITHOUT_RATING = ("## 최종 판단\n\n금리 경로가 분명해지기 전까지는 판단을 유보한다. "
                               "다음 공시를 확인한 뒤 다시 본다.")
_RATING_WORDS = ("buy", "overweight", "hold", "underweight", "sell")

# Whole-hour zones as POSIX TZ strings (no tzdata lookup, same on macOS and glibc) on either
# side of the date line: (TZ value, the same offset for computing that zone's date here).
# UTC-12 is a calendar day behind UTC until 12:00 UTC; UTC+14 a day ahead from 10:00 UTC.
_UTC_MINUS_12 = ("<-12>+12", datetime.timezone(datetime.timedelta(hours=-12)))
_UTC_PLUS_14 = ("<+14>-14", datetime.timezone(datetime.timedelta(hours=14)))


# =============================================================================== helpers

def _expected_plan(name: str) -> dict[str, Any]:
    """TRADE_PLAN_JSON the PM's typed plan must produce: its raw tool args minus the prose.

    s1's PM returned the synthetic Sell plan (fixtures/synthetic_sell_plan.json, a
    TRADE_PLAN_JSON payload) plus the revision it wrote.
    """
    if name == S1:
        return {**scenarios.sell_plan(), "revision": scenarios.REIT_REVISION}
    typed = {S4: scenarios.APPLE_OVERWEIGHT_PM, S5A: scenarios.SAMSUNG_BUY_PM}[name]
    return {k: v for k, v in typed.items() if k not in PLAN_PROSE_FIELDS}


def _quote_scenario(case: str) -> dict[str, Any]:
    """s1's holding and PM plan (no memory seed) with exactly one quoted rating added."""
    base = scenarios.get(S2)
    base["roles"]["portfolio_manager"]["structured"] = scenarios.reit_sell_pm()
    patch, _marker = QUOTE_CASES[case]
    return scenarios.derive(base, {"roles": {"portfolio_manager": {"structured": patch}}},
                            name=f"output_quote_{case}",
                            description=f"s1's Sell plan, PM prose quoting another rating: {case}")


def _fallback_scenario(trigger: str) -> dict[str, Any]:
    patch = FALLBACK_TRIGGERS[trigger]
    if patch is None:
        return scenarios.get(S3)
    return scenarios.derive(scenarios.get(S3), {"roles": {"portfolio_manager": patch}},
                            name=f"output_fallback_{trigger}",
                            description=f"s3 with the PM's structured call failing by: {trigger}")


def _fallback_without_rating_scenario() -> dict[str, Any]:
    """s3 (the PM's structured call fails) whose free-text answer names no rating."""
    return scenarios.derive(
        scenarios.get(S3), {"roles": {"portfolio_manager": {"text": PM_FREE_TEXT_WITHOUT_RATING}}},
        name="output_fallback_without_a_rating",
        description="s3 with a PM fallback answer that defers the call and names no rating")


def _zone_off_the_utc_date() -> tuple[str, datetime.timezone]:
    """A zone whose calendar date is not the UTC date for at least the next hour, except
    within an hour of UTC midnight (the caller detects a run that straddles it)."""
    hour = datetime.datetime.now(datetime.timezone.utc).hour
    return _UTC_MINUS_12 if hour < 11 else _UTC_PLUS_14


def _last_memory_tag(res: RunResult) -> dict[str, str]:
    """{date, ticker, rating} of the last entry tag in this run's per-ticker memory log."""
    log = res.memory_log or ""
    tags = [m.groupdict() for m in map(_MEMORY_TAG_RE.match, log.splitlines()) if m]
    assert tags, f"no entry tag in the memory log {res.memory_log_path}:\n{log[:500]}"
    return tags[-1]


def _writer_sections(markdown: str) -> dict[str, str]:
    """{writer heading: text up to the next writer heading}; asserts presence, uniqueness, order."""
    lines = markdown.splitlines()
    positions = []
    for heading in WRITER_HEADINGS:
        found = [i for i, line in enumerate(lines) if line == heading]
        assert len(found) == 1, f"{heading!r} appears {len(found)} times in complete_report.md"
        positions.append(found[0])
    actual_order = [WRITER_HEADINGS[positions.index(p)] for p in sorted(positions)]
    assert actual_order == WRITER_HEADINGS, actual_order
    bounds = [*positions, len(lines)]
    return {heading: "\n".join(lines[bounds[i] + 1: bounds[i + 1]]).strip()
            for i, heading in enumerate(WRITER_HEADINGS)}


def _model_table(markdown: str) -> dict[str, tuple[str, str]]:
    """Role -> (provider, model) rows of the header table (before the first section)."""
    header = markdown.split("\n## I. Analyst Team Reports", 1)[0]
    rows = [m for m in map(_MODEL_ROW_RE.match, header.splitlines()) if m]
    return {m["role"]: (m["provider"], m["model"]) for m in rows}


# =============================================================================== exit code

@pytest.mark.parametrize("name", [S1, S3, S4, S5A])
def test_main_py_exits_zero_without_a_traceback(ap_run, name):
    """Breaks if: anything main.py runs raises — a post-run import a merge moved or deleted
    (``cli.main.save_report_to_disk``, ``tradingagents.dataflows.utils``), a KR data module
    whose import broke (``from .symbol_utils import NoMarketDataError``), the report
    writer's own lazy imports. alpha-pulse marks an rc != 0 run failed and discards its plan;
    the post-run failures happen after every LLM call of the run was already paid for."""
    res = ap_run(name)
    res.assert_ok()
    assert "Traceback (most recent call last)" not in res.stderr, res.stderr[-3000:]
    assert (res.captures.get("exit") or {}).get("kind") == "returned", res.describe()


# =============================================================================== stdout grammar

@pytest.mark.parametrize("name", list(STRUCTURED_RUNS))
def test_stdout_ends_with_the_decision_the_plan_and_the_report_line(ap_run, name):
    """Breaks if: main.py changes how stdout closes — prints the decision in another form (an
    Enum repr like 'PortfolioRating.SELL' on Python 3.11, a sentence), drops or repeats the
    TRADE_PLAN_JSON line, prints anything after 'Report saved:', or saves the report
    anywhere but ``$TRADINGAGENTS_RESULTS_DIR/reports/<TICKER>_<YYYYmmdd_HHMMSS>/``.
    alpha-pulse takes the decision from the last bare rating line before the last
    'Report saved:' line, so a changed decision line silently falls back to an older rating
    word in the debug trace, and a report outside the results dir is not where alpha-pulse
    keeps it."""
    typed, ticker, _label = STRUCTURED_RUNS[name]
    res = ap_run(name).assert_ok()
    decision_line, plan_line, report_line = res.tail
    assert decision_line == typed, res.tail
    assert plan_line.startswith("TRADE_PLAN_JSON: {"), plan_line[:120]
    plan_like = [ln for ln in res.stdout_lines if "TRADE_PLAN_JSON" in ln]
    assert plan_like == [plan_line], [ln[:80] for ln in plan_like]
    m = REPORT_SAVED_LINE_RE.match(report_line)
    assert m, report_line
    report = Path(m["path"])
    assert report.is_absolute(), report
    assert report.parent.parent == res.results_dir / "reports", report
    assert m["dir"] == ticker
    datetime.datetime.strptime(m["stamp"], "%Y%m%d_%H%M%S")  # a real wall-clock stamp
    assert report.is_file(), report
    assert contract_reader.read_stdout(res.stdout) == (typed, m["path"])


@pytest.mark.parametrize("name", list(STRUCTURED_RUNS))
def test_trade_plan_json_is_the_typed_plan_without_its_prose(ap_run, name):
    """Breaks if: the TRADE_PLAN_JSON payload stops carrying the Portfolio Manager's typed plan
    under the keys alpha-pulse reads — a key renamed or dropped (``revision`` included:
    alpha-pulse's revision gate then refuses the plan; ``exclude_none`` drops every null
    field, and a kill switch without its ``price`` key is rejected by alpha-pulse), a value
    changed or re-typed (11120.0 printed as 11120 or "11120"), the two prose fields printed
    on the line although the documented line format excludes them (main.py:
    ``model_dump_json(exclude={"executive_summary","investment_thesis"})``), the typed
    object no longer threaded through the graph state (no line at all: langgraph drops
    undeclared state keys silently), or the JSON spread over several lines (a one-line
    reader then finds nothing). Each of these ends as "no plan" (plan_status 'unparsed'), a
    refused plan or wrong order drafts. A NEW optional key is not a break: alpha-pulse reads
    the plan by key and ignores keys it does not know, so the payload is compared on the
    keys the plan has today (a new REQUIRED schema field is the trade-plan module's concern:
    it drops the whole plan)."""
    res = ap_run(name).assert_ok()
    plan = contract_reader.read_trade_plan(res.stdout)
    assert plan is not None, res.describe()
    expected = _expected_plan(name)
    assert canonical_json(project_to_expected(plan, expected)) == canonical_json(expected)
    assert sorted(set(PLAN_PROSE_FIELDS) & set(plan)) == []


def test_sell_plan_reads_back_with_its_hand_checked_numbers(ap_run):
    """Breaks if: the Sell plan's numbers or vocabulary change on the way out — the immediate
    tranche band, the stop, a take-profit's order price kept apart from the level it watches,
    the trailing percent, the trigger kinds of each tranche, the full exit, the kill switch
    and the revision that alpha-pulse's revision gate reads. Literals copied by hand from the
    synthetic Sell fixture (fixtures/synthetic_sell_plan.json) and s1's revision, not
    derived from any code. Nested objects are compared on the keys listed here: a new
    optional key alpha-pulse does not read is not a break."""
    plan = ap_run(S1).assert_ok().plan
    assert plan is not None
    head = (plan["rating"], plan["price_target"], plan["total_weight_pct"], plan["stop_loss"])
    assert head == ("Sell", 11120.0, None, 10380.0)
    first_tranche = {"seq": 1, "pct": 50.0, "price_low": 10720.0, "price_high": 10860.0,
                     "trigger": "immediate", "triggers": [], "condition": None}
    take_profit = {"kind": "take_profit", "price": 11060.0, "trail_pct": None,
                   "reference_price": 11090.0, "reference_label": "직전 고점"}
    trailing = {"kind": "trailing", "price": None, "trail_pct": 4.0, "reference_price": None}
    full_exit = {"kind": "full", "remaining_weight_pct": None}
    revision = {"kind": "new_information",
                "note": "차입금 차환 금리가 진입 당시 가정보다 높게 정해져, 기대한 배당 여력이 "
                        "줄었다."}
    assert [(t["seq"], t["pct"], t["trigger"]) for t in plan["tranches"]] == \
        [(1, 50.0, "immediate"), (2, 30.0, "conditional"), (3, 20.0, "conditional")]
    assert [[trg["kind"] for trg in t["triggers"]] for t in plan["tranches"]] == \
        [[], ["take_profit", "trailing", "event"], ["stop", "take_profit", "event"]]
    second_triggers = plan["tranches"][1]["triggers"]
    for actual, expected in ((plan["tranches"][0], first_tranche),
                             (second_triggers[0], take_profit), (second_triggers[1], trailing),
                             (plan["exit_target"], full_exit), (plan["revision"], revision)):
        assert canonical_json(project_to_expected(actual, expected)) == canonical_json(expected)
    assert plan["kill_switch"]["price"] == 10290.0


def test_last_line_grammar_skips_decoy_lines_in_the_debug_trace(ap_run):
    """Breaks if: main.py stops closing stdout with its own decision / 'Report saved:' lines.
    main.py runs the graph with debug=True, so agent prose is printed to stdout first; in
    this run the market analyst wrote a bare 'Hold' line and a fake 'Report saved:' line.
    The documented last-line grammar must still yield the typed 'Sell' and the real report
    path — and the decoys are live: without main.py's three closing lines, the same grammar
    reads them."""
    res = ap_run(S1).assert_ok()
    lines = res.stdout_lines
    decoy_report = "Report saved: /tmp/decoy/complete_report.md"
    assert lines.count(decoy_report) == 1, "the debug trace no longer carries the decoy"
    decoy_at = lines.index(decoy_report)
    assert decoy_at < len(lines) - 3, "the decoy must sit in the trace, before main.py's lines"
    assert lines[decoy_at - 1] == "Hold", lines[decoy_at - 2: decoy_at + 1]
    real_path = res.tail[-1][len("Report saved: "):]
    assert contract_reader.read_stdout(res.stdout) == ("Sell", real_path)
    # control: the same trace without main.py's closing lines reads as the decoys
    trace_only = "\n".join(lines[:-3]) + "\n"
    assert contract_reader.read_stdout(trace_only) == ("Hold", "/tmp/decoy/complete_report.md")


@pytest.mark.parametrize("case", list(TYPED_UNDER_QUOTES))
def test_decision_plan_and_memory_tag_follow_the_typed_rating_when_prose_quotes_other_ratings(
        ap_run, case):
    """Breaks if: the rating read back from the PM's rendered decision stops being the typed
    one — upstream v0.5.1's 'last rating label wins' parser (8d30fee) turns a typed Sell into
    the Buy / Overweight / Hold the prose quotes, on the stdout decision line AND in the
    memory-log tag (both use that parser). alpha-pulse shows the decision line as the run's
    decision and the next same-ticker runs read the tag as the past call, while orders
    follow plan.rating — the two then disagree silently at rc 0."""
    typed = TYPED_UNDER_QUOTES[case]
    if case in QUOTE_CASES:
        spec, markers = _quote_scenario(case), [QUOTE_CASES[case][1]]
    else:  # the ready scenarios carry all four quotes
        spec, markers = scenarios.get(case), [marker for _, marker in QUOTE_CASES.values()]
    res = ap_run(spec).assert_ok()
    # Armed: the quote reached the rendered decision that the rating parser reads.
    rendered = (res.final_state or {}).get("final_trade_decision") or ""
    for marker in markers:
        assert marker in rendered, (marker, rendered[:300])
    assert res.tail[0] == typed, res.tail[0]
    assert res.decision == typed
    assert (res.plan or {}).get("rating") == typed
    assert _last_memory_tag(res) == {"date": TRADE_DATE, "ticker": "417310.KS", "rating": typed}


@pytest.mark.parametrize("name", list(STRUCTURED_RUNS))
def test_memory_log_tag_records_the_typed_rating_of_the_run(ap_run, name):
    """Breaks if: the per-ticker memory-log entry this run appends stops carrying the typed
    rating (or the ISO trade date / the ticker as passed) — the next same-ticker runs read
    that tag back to the Portfolio Manager as the call made on this date."""
    typed, ticker, _label = STRUCTURED_RUNS[name]
    res = ap_run(name).assert_ok()
    assert _last_memory_tag(res) == {"date": TRADE_DATE, "ticker": ticker, "rating": typed}


@pytest.mark.parametrize("trigger", list(FALLBACK_TRIGGERS))
def test_structured_failure_prints_no_plan_but_a_decision_and_the_full_report(ap_run, trigger):
    """Breaks if: the Portfolio Manager's free-text fallback stops being a clean 'no plan' run
    for any way the structured call fails in production (the model answers in prose, the
    provider errors, a plan number the schema cannot read such as '10,380원'): it crashes
    (rc 1), prints a TRADE_PLAN_JSON anyway (upstream's lenient ``_coerce_optional_float``
    reads '10,380원' as a null stop and prints a plan alpha-pulse drafts orders from, instead
    of storing "no plan" — plan_status 'unparsed'), drops the decision line (the last-line
    grammar would then take a rating word from the debug trace), or the writer loses the PM
    section and decision.md."""
    res = ap_run(_fallback_scenario(trigger)).assert_ok()
    assert [ln for ln in res.stdout_lines if "TRADE_PLAN_JSON" in ln] == []
    assert contract_reader.read_trade_plan(res.stdout) is None
    decision_line, report_line = res.tail[-2:]
    assert decision_line.strip().lower() in DECISION_WORDS, res.tail  # REVIEW is allowed
    m = REPORT_SAVED_LINE_RE.match(report_line)
    assert m and Path(m["path"]).is_file(), report_line
    assert contract_reader.read_stdout(res.stdout) == (decision_line.strip(), m["path"])
    assert set(REQUIRED_REPORT_FILES) <= set(res.report_files), res.report_files
    assert scenarios.PM_FREE_TEXT in res.report_file("5_portfolio/decision.md")
    assert scenarios.PM_FREE_TEXT in _writer_sections(res.complete_report)["### Portfolio Manager"]


def test_decision_without_any_rating_reaches_alphapulse_as_review_not_as_a_tradeable_word(ap_run):
    """Breaks if: a Portfolio Manager decision that names no rating stops reaching alpha-pulse
    as the REVIEW sentinel — reverting 3eb5f43 (SignalProcessor falling back to a fabricated
    'Hold'), a rating parser whose default is a rating, or main.py printing another line for
    it. alpha-pulse keeps the decision line as the run's decision and treats REVIEW as "do not
    act on this, look at it"; a 'Hold' in its place is a tradeable neutral call on its
    dashboard, in its notifications and in the discovery report, so a failed judgement passes
    as a real one at rc 0 (there is no plan either way, so nothing else shows it)."""
    res = ap_run(_fallback_without_rating_scenario()).assert_ok()
    # Armed: the structured call failed and the decision text is the rating-free fallback.
    assert res.calls_for("portfolio_manager", "structured"), sorted(res.roles_called())
    rendered = (res.final_state or {}).get("final_trade_decision") or ""
    assert PM_FREE_TEXT_WITHOUT_RATING in rendered, rendered[:300]
    assert [w for w in _RATING_WORDS if w in rendered.lower()] == [], rendered[:300]

    assert res.trade_plan_lines == []
    decision_line, report_line = res.tail[-2:]
    assert REPORT_SAVED_LINE_RE.match(report_line), res.tail
    assert contract_reader.is_review(decision_line), res.tail
    # The run as a whole: completed (rc 0, the report line read), no plan, and the decision
    # read back is the REVIEW sentinel -- not one of the tradeable words.
    assert (res.rc, res.plan) == (0, None), res.describe()
    assert res.report_path == report_line[len("Report saved: "):], res.tail
    assert contract_reader.is_review(res.decision), res.decision
    assert (res.decision or "").strip().lower() not in contract_reader.TRADEABLE_WORDS


@pytest.mark.parametrize(("name", "decision", "has_plan"), [
    (S1, "Sell", True),          # held on a Buy thesis, typed Sell with a revision
    (S4, "Overweight", True),    # discovery: no account context
    (S3, None, False),           # no plan line; the decision is the fallback's (REVIEW allowed)
])
def test_alphapulse_records_the_run_as_done_with_a_readable_report(ap_run, name, decision,
                                                                   has_plan):
    """Breaks if: what alpha-pulse stores for a successful run changes — the run reads as
    failed (rc != 0 or no 'Report saved:' line), another decision, a plan that appears or
    vanishes (or whose rating is not the decision line's), or a report directory (the parent
    of the printed path) whose complete_report.md its web view and discovery cannot open."""
    res = ap_run(name)
    assert res.rc == 0 and res.report_path, res.describe()
    if decision is None:
        assert (res.decision or "").lower() in DECISION_WORDS, res.decision
    else:
        assert res.decision == decision
    assert (res.plan is not None) == has_plan, res.trade_plan_lines
    if has_plan:
        assert res.plan["rating"] == res.decision, (res.plan["rating"], res.decision)
    report = Path(res.report_path).parent / "complete_report.md"
    assert report.is_file(), report
    assert report.read_text(encoding="utf-8") == res.complete_report


# =============================================================================== report tree

@pytest.mark.parametrize("name", [S1, S3, S4])
def test_report_directory_holds_every_section_file(ap_run, name):
    """Breaks if: a section file disappears or moves — e.g. upstream cf960d6 dropped the debate
    states' ``judge_decision``, so a writer still reading it loses 2_research/manager.md and
    5_portfolio/decision.md at rc 0."""
    res = ap_run(name).assert_ok()
    missing = sorted(set(REQUIRED_REPORT_FILES) - set(res.report_files))
    assert not missing, (missing, res.report_files)
    for rel in REQUIRED_REPORT_FILES:
        assert res.report_file(rel).strip(), f"{rel} is empty"


@pytest.mark.parametrize("name", list(STRUCTURED_RUNS))
def test_complete_report_title_names_the_company_and_the_ticker(ap_run, name):
    """Breaks if: the report title loses the company name — upstream's writer titles the
    report with the bare ticker, and a lost identity lookup does the same. The web view and
    the discovery deep report show this title."""
    label = STRUCTURED_RUNS[name][2]
    res = ap_run(name).assert_ok()
    assert res.complete_report.splitlines()[0] == f"# Trading Analysis Report: {label}"


def test_complete_report_header_lists_the_model_behind_every_role(ap_run):
    """Breaks if: the report header stops documenting the models — upstream's writers have no
    per-role table, and a merged build_config that loses role_models or the tiering lists
    other models (RM/PM must read claude-opus-5, the other ten roles claude-sonnet-5, all on
    vertex_anthropic)."""
    res = ap_run(S1).assert_ok()
    table = _model_table(res.complete_report)
    assert {role: table.get(role) for role in ROLE_MODELS} == ROLE_MODELS, table


def test_complete_report_has_every_section_in_order_with_the_decisions_in_it(ap_run):
    """Breaks if: complete_report.md loses or reorders a section or carries an empty one —
    above all the Research Manager and Portfolio Manager decisions (a writer reading
    ``judge_decision`` after upstream cf960d6 drops both at rc 0). The PM section must state
    the typed rating and keep the model's own wording (account figures are scrubbed from
    the memory log only, never from the report)."""
    res = ap_run(S1).assert_ok()
    sections = _writer_sections(res.complete_report)
    story = scenarios.REIT_STORY
    expected_text = {
        "### Market Analyst": story["market"],
        "### Sentiment Analyst": story["sentiment"],
        "### News Analyst": story["news"],
        "### Fundamentals Analyst": story["fundamentals"],
        "### Bull Researcher": story["bull"],
        "### Bear Researcher": story["bear"],
        "### Research Manager": scenarios.REIT_RESEARCH["rationale"],
        "### Trader": scenarios.REIT_TRADER["reasoning"],
        "### Aggressive Analyst": story["aggressive"],
        "### Conservative Analyst": story["conservative"],
        "### Neutral Analyst": story["neutral"],
        "### Portfolio Manager": scenarios.REIT_SELL_EXEC_SUMMARY,
    }
    for heading, text in expected_text.items():
        assert text in sections[heading], f"{heading}: {sections[heading][:300]!r}"
    assert "Underweight" in sections["### Research Manager"]
    pm = sections["### Portfolio Manager"]
    assert re.search(r"^\*\*Rating\*\*: Sell$", pm, re.M), pm[:300]
    assert "421,675,870원" in pm
    assert pm.startswith(res.report_file("5_portfolio/decision.md").strip())


# =============================================================================== dates

@pytest.mark.parametrize(("name", "argv_date"), [(S1, "20260819"), (S5A, "20260819"),
                                                 (S4, "2026-08-19")])
def test_argv_date_reaches_propagate_as_an_iso_date(ap_run, name, argv_date):
    """Breaks if: main.py stops normalising the date argument — the nightly batch passes the
    KST date as 'YYYYMMDD' (``strftime('%Y%m%d')``); a strict YYYY-MM-DD parser
    exits 2 before any analysis, and handing the raw '20260819' to the graph fails inside it
    (the fork's sentiment node, upstream's trade-date validation: rc 1)."""
    res = ap_run(name)
    assert res.argv[1] == argv_date
    res.assert_ok()
    assert res.trade_date_seen == TRADE_DATE
    assert (res.final_state or {}).get("trade_date") == TRADE_DATE
    assert _last_memory_tag(res)["date"] == TRADE_DATE


def test_missing_date_defaults_to_today_in_kst(tmp_path):
    """Breaks if: main.py stops defaulting a missing date to today's date as the process sees
    it under TZ=Asia/Seoul (alpha-pulse appends no date when trade_date is empty). Both KST
    dates around the run are accepted (midnight). A UTC 'today' differs from it only between
    00:00 and 09:00 KST; the next test catches that regression at any hour."""
    before = kst_today()
    res = run_main(tmp_path, S5B)
    after = kst_today()
    res.assert_ok()
    assert res.argv == ["005930.KS"]
    assert res.env_seen.get("TZ") == "Asia/Seoul"
    assert res.trade_date_seen in {before.isoformat(), after.isoformat()}, res.trade_date_seen
    assert (res.final_state or {}).get("trade_date") == res.trade_date_seen
    assert _last_memory_tag(res)["date"] == res.trade_date_seen


def test_missing_date_follows_the_process_clock_when_utc_is_on_another_day(tmp_path):
    """Breaks if: main.py's missing-date default stops being the date of the process's own
    clock, which alpha-pulse sets with TZ (it runs main.py under TZ=Asia/Seoul) — e.g. a UTC
    'today' (``datetime.now(timezone.utc).date()``, ``utcnow()``): every no-date run started
    between 00:00 and 09:00 KST would then analyse yesterday. The KST run above can tell the
    two apart only in that window; this run is deterministic at any hour because it moves
    the process clock instead of waiting for the window — it runs under a whole-hour zone
    whose date is not the UTC date throughout the run (UTC-12 before 11:00 UTC, UTC+14
    after), and a run that straddles UTC midnight is repeated once. (A main.py that pinned
    Asia/Seoul itself would satisfy production but not this run: alpha-pulse's lever is TZ.)"""
    utc = datetime.timezone.utc
    zone = ""
    local_dates: set[datetime.date] = set()
    utc_dates: set[datetime.date] = set()
    res: RunResult | None = None
    for attempt in range(2):
        zone, offset = _zone_off_the_utc_date()
        before = (datetime.datetime.now(offset).date(), datetime.datetime.now(utc).date())
        spec = scenarios.derive(scenarios.get(S5B), {}, name=f"s5b_no_date_off_utc_{attempt}")
        res = run_main(tmp_path, spec, env_overrides={"TZ": zone})
        after = (datetime.datetime.now(offset).date(), datetime.datetime.now(utc).date())
        local_dates, utc_dates = {before[0], after[0]}, {before[1], after[1]}
        if not local_dates & utc_dates:
            break  # armed: no date the process could read is a UTC date
    else:
        pytest.fail(f"not armed: under TZ={zone} the process date {sorted(map(str, local_dates))} "
                    f"was a UTC date {sorted(map(str, utc_dates))} around both runs -- each run "
                    "straddled UTC midnight (rerun), or the zone choice no longer separates the "
                    "dates (fix _zone_off_the_utc_date); a pass here would prove nothing")
    assert res is not None
    res.assert_ok()
    assert res.argv == ["005930.KS"]
    assert res.env_seen.get("TZ") == zone
    assert res.trade_date_seen in {d.isoformat() for d in local_dates}, (
        f"TZ={zone}: main.py analysed {res.trade_date_seen}; the process date was "
        f"{sorted(map(str, local_dates))}, the UTC date {sorted(map(str, utc_dates))}")
    assert (res.final_state or {}).get("trade_date") == res.trade_date_seen
    assert _last_memory_tag(res)["date"] == res.trade_date_seen


# =============================================================================== imports

def test_every_import_in_main_py_resolves_from_this_checkout(tmp_path):
    """Breaks if: an import main.py performs stops resolving — above all the ones main() runs
    only after propagate returned (``from cli.main import save_report_to_disk``, ``from
    tradingagents.dataflows.utils import safe_ticker_component``), which upstream moved or
    removed (dataflows/utils.py -> dataflows/symbols.py, cli/utils.py deleted, no
    save_report_to_disk). Production then exits 1 after every LLM call of the run was paid
    for; the full runs above fail too, this test names the missing module and name."""
    run_dir = tmp_path / "main_py_imports"
    for sub in ("home", "tmp", "results", "cache", "memory", "fake_app"):
        (run_dir / sub).mkdir(parents=True)
    helper = PACKAGE_DIR / "_output_helpers.py"
    assert helper.is_file()
    proc = subprocess.run(
        [harness_python(), "-m", "tests.alphapulse_contract._output_helpers", str(REPO_ROOT),
         str(run_dir)],
        cwd=str(REPO_ROOT), env=base_env(run_dir, "IMPORTS"), capture_output=True, timeout=300)
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")[-3000:]
    records = json.loads((run_dir / "imports.json").read_text(encoding="utf-8"))

    def in_fork(module: str | None) -> bool:
        return (module or "").split(".")[0] in ("tradingagents", "cli")

    def where(r: dict[str, Any]) -> str:
        when = "after propagate" if r["after_propagate"] else "before propagate"
        stmt = (f"import {r['module']}" if r["name"] is None
                else f"from {'.' * r['level']}{r['module'] or ''} import {r['name']}")
        return f"main.py:{r['line']} in {r['function'] or '<module>'} ({when}): {stmt}"

    fork = [r for r in records if in_fork(r["module"])]
    assert fork, f"main.py imports nothing from the fork? {records}"
    failures = [f"{where(r)} -> {r['error']}" for r in records if not r["ok"] and not r["optional"]]
    assert not failures, "\n".join(failures)
    repo = REPO_ROOT.resolve()
    for r in fork:  # resolved from THIS checkout, not an installed or editable copy elsewhere
        files = [r["module_file"]] + ([r["object_file"]] if in_fork(r.get("object_module")) else [])
        for file in files:
            assert Path(file).resolve().is_relative_to(repo), (where(r), file)
