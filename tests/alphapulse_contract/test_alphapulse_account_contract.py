"""Account-context contract: alpha-pulse's account snapshot reaches the Portfolio Manager only.

alpha-pulse runs ``main.py TICKER [DATE]`` with ``TRADINGAGENTS_POSITION_CONTEXT`` set to

* its one-line account JSON (docs/INTEGRATION.md §4; nightly batch and web runs): the
  nightly batch adds the ``founding_thesis`` of the plan that opened the position, the web
  entry points send ``founding_thesis_absent_reason: out_of_scope``; or
* the explicit empty string (discovery deep runs, and any run whose account read failed):
  "not injected", deliberately SET so that a stale line in a cwd ``.env`` cannot leak in
  (a silent wrong injection is worse than a silent missing one).

What alpha-pulse relies on, frozen here through the real ``main.py`` (``harness.run_main``;
only the LLM and the data libraries are faked):

1. The Portfolio Manager's prompt carries the account block -- holdings, the Founding
   Thesis (or why it is absent) and the instruction to fill ``revision`` -- exactly as
   a2981a7 renders it. alpha-pulse's revision gate assumes the PM SAW that thesis whenever
   it sent one (the fork's ``PlanRevision`` docstring states that contract).
2. No other LLM call sees anything of the account. Analysts, Bull/Bear, Research Manager,
   Trader, the three risk debators and the settlement Reflector get the same prompts
   whether the account was sent or not, and no prompt carries upstream's
   "Portfolio context: not provided ... do not assume a flat book" notice.
3. ``''`` (and a blank value) means not injected: the PM prompt has no account section.
4. The account survives to the final state, and the archived memory-log copy of the
   decision has every account figure redacted (the archive is re-injected into the next
   five same-ticker PM prompts) while the saved report keeps the model's wording.
5. An explicit ``''`` beats a ``TRADINGAGENTS_POSITION_CONTEXT`` line in the cwd ``.env``.
6. ``main.py`` hands ``propagate`` the ticker and the date only -- no ``portfolio=``
   (upstream's caller-portfolio channel is read by the Trader and all three risk debators).

Reviewed fixtures: ``fixtures/position_block_holding_founding_buy.txt`` and
``fixtures/position_block_not_held.txt`` are ``build_position_block`` output captured at
a2981a7 for the two synthetic account JSONs in ``fixtures/position_contexts.json``
(``holding_founding_buy``, ``not_held``), reviewed line by line against the renderer
(holding line, last price, P&L, weight, cash, NAV, Founding Thesis lines, the ``revision``
instruction, the do-not-quote instruction; ``\\n``-joined with one trailing ``\\n``).
They change only together with alpha-pulse's Founding Thesis / revision contract.
"""

from __future__ import annotations

import difflib
import inspect
import json
import re
from pathlib import Path
from typing import Any

import pytest

from . import scenarios
from ._compat import get_symbol
from ._runtime import json_safe
from .harness import FIXTURES, REPO_ROOT, RunResult

S1 = "s1_nightly_kr_holding_sell"
S5A_NOT_HELD = "s5a_nightly_yyyymmdd_not_held"
TICKER = "417310.KS"
ISO_TRADE_DATE = "2026-08-19"  # s1 argv passes the nightly KST form '20260819'
BUILD_POSITION_BLOCK_HOMES = (
    "tradingagents.agents.utils.agent_utils",  # a2981a7
    "tradingagents.agents.context",            # upstream v0.5.x layout
)

HOLDING_JSON = scenarios.position_context("holding_founding_buy")
BLANK_ENV_VALUES = {"explicit_empty": "", "whitespace": " \n\t "}

# Hand-written pieces of the reviewed holding block, for the absent-thesis variants below
# (test_a_held_position_without_founding_thesis_tells_the_pm_why checks they agree with it).
HELD_ACCOUNT_LINES = (
    "**Current Position and Account**\n"
    "- Holding: 1250 shares at an average cost of 10480.0 KRW\n"
    "- Last price: 10800.0 KRW\n"
    "- Unrealised P&L: 3.05%\n"
    "- Current weight: 2.99% of NAV\n"
    "- Cash available: 421675870.0 KRW\n"
    "- Total account NAV: 451015870.0 KRW\n"
)
DO_NOT_QUOTE_LINE = (
    "Use these figures to size and direct the decision. **Do not quote the raw account numbers "
    "in your written output** -- refer to sizing in percentages of NAV. The written decision is "
    "archived and reused as context on later runs, where these balances will be stale.\n"
)
ABSENT_THESIS_LINES = {  # founding_thesis_absent_reason values for a HELD position
    "lookup_failed": "- Founding thesis: **could not be retrieved** (infrastructure issue, not absence).\n",
    "no_source_plan": "- Founding thesis: none on record -- this position was not opened from a plan.\n",
    "out_of_scope": "",  # web / discovery entry points: holdings shown, no thesis line
}

# The account figures of fixtures/position_contexts.json 'holding_founding_buy'
# (held_qty 1250, avg_price 10480.0, cash 421675870.0, total_nav 451015870.0) in every
# textual shape a model writes them: grouped, bare integer, and the JSON float repr.
RAW_ACCOUNT_FIGURES = (
    "1,250", "1250",
    "10,480", "10480.0", "10480",
    "421,675,870", "421675870.0", "421675870",
    "451,015,870", "451015870.0", "451015870",
)
# Strings that exist only because the account was injected (none of them occurs in any
# prompt of the same run without the account; the blank-run tests re-check that).
ACCOUNT_MARKERS = (
    "Current Position and Account", "Founding Thesis", "Cash available", "Total account NAV",
    "1250 shares", "421675870", "421,675,870", "451015870", "451,015,870",
    "On 20260805 you rated this", "you MUST fill the `revision` field",
)
# Upstream 6436d1f: "Portfolio context: not provided. You do not know the caller's current
# holdings or cash, so do not assume a flat book; give direction and sizing guidance in terms
# the caller can apply to their own position." Several independent phrases, so a reworded
# notice that drops the two headline markers is still caught.
NOTICE_MARKERS = ("portfolio context", "flat book", "caller's current", "the caller can apply")

# The PM quotes the account in every shape (models do, despite the prompt): the archive
# copy must lose all of them, the saved report must keep them.
EXEC_SUMMARY_QUOTING_THE_ACCOUNT = (
    scenarios.REIT_SELL_EXEC_SUMMARY
    + " 계좌 원자료: held_qty 1250, avg_price 10480.0 (10480), cash 421675870.0 (421675870), "
      "total_nav 451,015,870원 (451015870.0 / 451015870)."
)
REPORT_KEEPS = ("1,250", "10,480", "10480.0", "421,675,870", "421675870.0", "451,015,870",
                "451015870")
STALE_DOTENV_ACCOUNT = json.dumps({
    "held_qty": 7777, "avg_price": 5555.0, "current_price": 5600.0, "unrealized_pnl_pct": 0.81,
    "current_weight_pct": 9.13, "cash": 98765432.0, "total_nav": 123456789.0, "currency": "KRW",
    "founding_thesis": {"as_of": "20260101", "rating": "Buy", "price_target": 7000.0,
                        "time_horizon": "6-12 months", "kill_switch_price": 4900.0,
                        "entry_price": 5555.0, "entry_date": "20260102", "trading_days_held": 150,
                        "last_night_rating": "Buy"},
    "founding_thesis_absent_reason": "",
}, ensure_ascii=False)
STALE_DOTENV_MARKERS = ("7777 shares", "7,777", "5555.0", "98765432", "98,765,432", "123456789",
                        "123,456,789", "20260101", "Current Position and Account",
                        "Founding Thesis")
DOTENV_PROBE = ("ALPHAPULSE_CONTRACT_DOTENV_PROBE", "loaded-from-the-cwd-dotenv")

_ACCT_BASE = scenarios.derive(
    scenarios.get(S1),
    {"roles": {"portfolio_manager": {"structured": {
        "executive_summary": EXEC_SUMMARY_QUOTING_THE_ACCOUNT}}}},
    description="s1 (nightly KR holding, Founding Thesis Buy, seeded memory log) whose PM prose "
                "quotes every account figure in grouped/bare/float form.",
)
ACCT_JSON = scenarios.derive(_ACCT_BASE, {}, name="acct_json", position_context=HOLDING_JSON)
ACCT_BLANK = {
    variant: scenarios.derive(_ACCT_BASE, {}, name=f"acct_{variant}", position_context=value)
    for variant, value in BLANK_ENV_VALUES.items()
}
ACCT_DOTENV = scenarios.derive(
    _ACCT_BASE, {}, name="acct_explicit_empty_vs_dotenv", position_context="",
    dotenv=(f"{DOTENV_PROBE[0]}={DOTENV_PROBE[1]}\n"
            f"TRADINGAGENTS_POSITION_CONTEXT='{STALE_DOTENV_ACCOUNT}'\n"),
)

NON_PM_ROLES = frozenset({
    "market_analyst", "sentiment_analyst", "news_analyst", "fundamentals_analyst",
    "bull_researcher", "bear_researcher", "research_manager", "trader",
    "aggressive_debator", "conservative_debator", "neutral_debator", "reflector",
})

_WALL_CLOCK = re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?")
_ENTRY_END = "<!-- ENTRY_END -->"


# ============================================================================ helpers

def _fork_symbol(name: str, *modules: str) -> Any:
    """``get_symbol`` that also proves the symbol comes from THIS checkout."""
    obj = get_symbol(name, *modules)
    path = Path(inspect.getfile(obj)).resolve()
    assert path.is_relative_to(REPO_ROOT.resolve()), (
        f"{name} was imported from {path}, not from the checkout under test ({REPO_ROOT}); "
        "an editable install of another checkout is shadowing it")
    return obj


def _reviewed_block(name: str) -> str:
    return (FIXTURES / f"position_block_{name}.txt").read_text(encoding="utf-8")


def holding_json_without_thesis(reason: str) -> str:
    """The 417310 holding's account JSON (position_contexts.json 'holding_founding_buy') as sent
    for a held position whose Founding Thesis could not be attached: ``founding_thesis`` null,
    ``founding_thesis_absent_reason`` = ``reason``; every other key, its order and every number's
    spelling unchanged."""
    data = json.loads(HOLDING_JSON)
    assert data["founding_thesis"] is not None, "premise: the fixture holding carries a thesis"
    data["founding_thesis"] = None
    data["founding_thesis_absent_reason"] = reason
    return json.dumps(data, ensure_ascii=False)


def _notice_hits(text: str) -> list[str]:
    low = text.lower()
    return [m for m in NOTICE_MARKERS if m in low]


def _normalized(res: RunResult, text: str) -> str:
    """Drop what legitimately differs between two runs: wall-clock stamps, the run dir."""
    return _WALL_CLOCK.sub("<wall-clock>", text).replace(str(res.run_dir), "<run_dir>")


def _prompts_by_role(res: RunResult) -> dict[str, list[str]]:
    """Every LLM call except the Portfolio Manager's, per role, in call order."""
    out: dict[str, list[str]] = {}
    for call in res.llm_calls:
        role = str(call.get("role"))
        if role == "portfolio_manager":
            continue
        out.setdefault(role, []).append(_normalized(res, res.prompt_text(call)))
    return out


def _first_difference(left: list[str], right: list[str], limit: int = 40) -> str:
    for i, (a, b) in enumerate(zip(left, right, strict=False)):
        if a != b:
            diff = difflib.unified_diff(a.splitlines(), b.splitlines(), "with account",
                                        "without account", lineterm="", n=1)
            return f"call {i}:\n" + "\n".join(line[:240] for line in list(diff)[:limit])
    return f"call counts differ: {len(left)} vs {len(right)}"


def _memory_entries(log: str) -> list[str]:
    return [chunk.strip() for chunk in log.split(_ENTRY_END) if chunk.strip()]


# ============================================================================ the block

@pytest.mark.parametrize("name", ["holding_founding_buy", "not_held"])
def test_position_block_for_alphapulse_account_json_is_the_reviewed_literal(name):
    """Breaks if: the PM's account block is re-worded, re-ordered or loses a line -- an
    upstream rewrite of the renderer (agents/context.py) or a merge that drops the Founding
    Thesis lines, the ``revision`` instruction or the do-not-quote instruction. alpha-pulse's
    revision gate is only fair if the PM was shown exactly this."""
    build_position_block = _fork_symbol("build_position_block", *BUILD_POSITION_BLOCK_HOMES)
    rendered = build_position_block({"position_context": scenarios.position_context(name)})
    assert rendered == _reviewed_block(name)


@pytest.mark.parametrize("reason", sorted(ABSENT_THESIS_LINES))
def test_a_held_position_without_founding_thesis_tells_the_pm_why(reason):
    """Breaks if: a held position whose Founding Thesis alpha-pulse could not attach is
    rendered differently -- the 'could not be retrieved' (infrastructure) line and the
    'not opened from a plan' line merged, dropped or swapped (an outage would then read as
    a manual buy), or the web/discovery ``out_of_scope`` case gaining a thesis line. Input:
    the fixture holding's JSON with the thesis replaced by the absent reason."""
    reviewed = _reviewed_block("holding_founding_buy")
    assert reviewed.startswith(HELD_ACCOUNT_LINES) and reviewed.endswith(DO_NOT_QUOTE_LINE)
    build_position_block = _fork_symbol("build_position_block", *BUILD_POSITION_BLOCK_HOMES)
    env_value = holding_json_without_thesis(reason)
    assert build_position_block({"position_context": env_value}) == (
        HELD_ACCOUNT_LINES + ABSENT_THESIS_LINES[reason] + DO_NOT_QUOTE_LINE)


# ============================================================================ the PM sees it

@pytest.mark.parametrize("case", ["holding_founding_buy", "not_held"])
def test_pm_prompt_carries_the_account_block_once_and_no_portfolio_notice(ap_run, case):
    """Breaks if: the account never reaches the PM (``_run_graph`` stops passing
    ``position_context``, the state channel is dropped, the PM hunk is resolved to upstream's
    ``{portfolio_context}`` template), or the PM gets upstream's "Portfolio context: not
    provided ... do not assume a flat book" notice next to real holdings, or the account is
    rendered a second time through another channel (instrument/past context, a portfolio
    block). Nightly holdings (Founding Thesis Buy) and nightly candidates (not held) are both
    alpha-pulse inputs."""
    res = ap_run(ACCT_JSON if case == "holding_founding_buy" else S5A_NOT_HELD).assert_ok()
    block = _reviewed_block(case)
    pm = res.prompt_for("portfolio_manager")
    assert pm.count(block) == 1, (
        f"the PM prompt carries the reviewed account block {pm.count(block)} times\n"
        + pm[:3000])
    if case == "holding_founding_buy":
        assert "**Founding Thesis** -- the plan this position was opened on" in pm
        assert "you MUST fill the `revision` field" in pm
    else:
        assert "**not currently held**" in pm
        assert "Founding Thesis" not in pm
    assert _notice_hits(pm) == [], pm[:3000]
    outside = pm.replace(block, "", 1)
    leaked = [m for m in ACCOUNT_MARKERS if m in outside]
    assert leaked == [], f"account data outside the block (a second channel?): {leaked}"


@pytest.mark.parametrize("variant", sorted(BLANK_ENV_VALUES))
def test_blank_position_context_is_not_injected_and_the_block_is_the_only_difference(
        ap_run, variant):
    """Breaks if: an explicit ``''`` (discovery deep, failed account read) or a blank value is
    treated as injected (an "account unavailable" heading, a stale default); if the blank
    run's PM prompt carries upstream's "Portfolio context: not provided" notice (unresolved
    6436d1f / the 'A' resolution); or if the PM prompt changes by anything but the block when
    the account is present (a notice toggled by the account: the rejected 'Z2' resolution)."""
    injected = ap_run(ACCT_JSON).assert_ok()
    blank = ap_run(ACCT_BLANK[variant]).assert_ok()
    assert blank.env_seen.get("TRADINGAGENTS_POSITION_CONTEXT") == BLANK_ENV_VALUES[variant]
    pm = blank.prompt_for("portfolio_manager")
    leaked = [m for m in ACCOUNT_MARKERS if m in pm]
    assert leaked == [], f"blank TRADINGAGENTS_POSITION_CONTEXT still injected {leaked}"
    assert _notice_hits(pm) == [], pm[:3000]
    block = _reviewed_block("holding_founding_buy")
    with_account = injected.prompt_for("portfolio_manager")
    assert with_account.count(block) == 1
    assert with_account.replace(block, "", 1) == pm, _first_difference(
        [with_account.replace(block, "", 1)], [pm])


# ============================================================================ nobody else does

@pytest.mark.parametrize("variant", sorted(BLANK_ENV_VALUES))
def test_the_account_reaches_no_llm_call_but_the_portfolio_managers(ap_run, variant):
    """Breaks if: any non-PM LLM call sees the account -- the env JSON merged into
    upstream's caller-portfolio channel (read by the Trader and the three risk debators), a
    Trader/Research-Manager/analyst injection, the account leaking through a shared field
    (instrument context) -- or its prompt depends on whether an account was sent at all (a
    notice toggled by the account, the rejected 'Z2' resolution). The PM is the last node,
    so it is the only place the account cannot bias another agent."""
    injected = ap_run(ACCT_JSON).assert_ok()
    blank = ap_run(ACCT_BLANK[variant]).assert_ok()
    with_account, without_account = _prompts_by_role(injected), _prompts_by_role(blank)
    assert set(with_account) >= NON_PM_ROLES, sorted(with_account)
    assert set(with_account) == set(without_account)
    for role, prompts in with_account.items():
        for i, prompt in enumerate(prompts):
            leaked = [m for m in ACCOUNT_MARKERS if m in prompt]
            assert leaked == [], f"{role} call {i} carries account data {leaked}"
    for role in sorted(with_account):
        assert with_account[role] == without_account[role], (
            f"{role}'s prompt depends on the account:\n"
            + _first_difference(with_account[role], without_account[role]))


def test_no_llm_call_carries_the_upstream_portfolio_notice(ap_run):
    """Breaks if: upstream 6436d1f's "Portfolio context: not provided ... do not assume a flat
    book" notice reaches any prompt of an alpha-pulse run (it is rendered whenever no caller
    portfolio is passed, i.e. on every alpha-pulse run) -- it rewrites the Trader's and
    debators' prompts and contradicts the PM's own holdings block. The merge must neutralise
    it (the 'Z3' config gate switched off in main.build_config)."""
    for spec in (ACCT_JSON, ACCT_BLANK["explicit_empty"], S5A_NOT_HELD):
        res = ap_run(spec).assert_ok()
        for call in res.llm_calls:
            hits = _notice_hits(res.prompt_text(call))
            assert hits == [], (res.scenario.get("name"), call.get("role"), hits)


# ============================================================================ state and archive

def test_position_context_survives_to_the_final_state(ap_run):
    """Breaks if: ``position_context`` is no longer declared in the graph's state schema
    (langgraph silently drops undeclared keys: no error, the PM sees nothing, and the archive
    scrub -- which reads the account back from the final state -- becomes a no-op), or
    ``_run_graph`` stops seeding it from the environment."""
    res = ap_run(ACCT_JSON).assert_ok()
    final_state = res.final_state
    assert final_state is not None, "propagate returned no state"
    assert "position_context" in final_state, sorted(final_state)
    assert json.loads(final_state["position_context"]) == json.loads(HOLDING_JSON)


def test_every_archived_memory_log_entry_is_scrubbed(ap_run):
    """Breaks if: the decision reaches the per-ticker memory log with account figures -- the
    archive hunk resolved to upstream's ``record_decision`` (no scrub), an extra unscrubbed
    write on another record/store path, the scrub fed an empty context (lost state
    channel), or a number shape the scrub stops covering. The raw balances would ride into
    the next five same-ticker PM prompts. Figures: position_contexts.json
    'holding_founding_buy' (held_qty 1250, avg_price 10480.0, cash 421675870.0, total_nav
    451015870.0); the PM prose quotes each in grouped, bare and float form."""
    res = ap_run(ACCT_JSON).assert_ok()
    log = res.memory_log
    assert log, f"no memory log at {res.memory_log_path}"
    entries = _memory_entries(log)
    new = [e for e in entries if e.startswith(f"[{ISO_TRADE_DATE} | {TICKER} |")]
    assert len(new) == 1, [e.splitlines()[0] for e in entries]
    (entry,) = new
    assert "세 단계로 전량 처분한다" in entry, "the decision itself was not archived"
    assert "[redacted]" in entry, entry
    for i, archived in enumerate(entries):
        leaked = [f for f in RAW_ACCOUNT_FIGURES if f in archived]
        assert leaked == [], f"memory-log entry {i} archives account figures {leaked}"


def test_the_saved_report_keeps_the_pm_wording_unscrubbed(ap_run):
    """Breaks if: the redaction spills beyond the archive copy -- the scrub moved into the PM
    node, or otherwise applied to the decision text the report is written from -- so
    complete_report.md, which alpha-pulse's web view and discovery read, shows '[redacted]'
    instead of the model's own figures."""
    res = ap_run(ACCT_JSON).assert_ok()
    report = res.complete_report or ""
    assert "세 단계로 전량 처분한다" in report, "the PM decision is missing from the report"
    lost = [f for f in REPORT_KEEPS if f not in report]
    assert lost == [], f"the saved report lost the PM's own figures {lost}"


# ============================================================================ entry point

def test_explicit_empty_env_beats_a_position_context_in_the_cwd_dotenv(ap_run):
    """Breaks if: a ``TRADINGAGENTS_POSITION_CONTEXT`` line left in the cwd ``.env`` (main.py
    runs with cwd = the checkout) overrides the explicit ``''`` alpha-pulse sets for
    discovery runs -- ``load_dotenv(override=True)``, or an env reader that falls back to the
    .env when the value is empty -- so every discovery run would read one stale snapshot as
    its current holding."""
    res = ap_run(ACCT_DOTENV).assert_ok()
    reference = ap_run(ACCT_BLANK["explicit_empty"]).assert_ok()
    for call in res.llm_calls:
        leaked = [m for m in STALE_DOTENV_MARKERS if m in res.prompt_text(call)]
        assert leaked == [], (call.get("role"), leaked)
    assert res.prompt_for("portfolio_manager") == reference.prompt_for("portfolio_manager")
    seen = res.env_seen
    assert seen.get("TRADINGAGENTS_POSITION_CONTEXT") == ""
    # Precondition: the fork really loaded the served .env. If it no longer loads a cwd
    # .env this test is vacuous -- rethink how a stale value could reach a run, don't delete.
    assert seen.get(DOTENV_PROBE[0]) == DOTENV_PROBE[1], (
        "the fork no longer loads the cwd .env", res.captures.get("dotenv"))


def test_main_hands_propagate_only_the_ticker_and_the_iso_date(ap_run):
    """Breaks if: main.py starts passing anything else to ``propagate`` -- above all a
    ``portfolio=`` built from the account (upstream's caller-portfolio channel renders into
    the Trader's and all three risk debators' prompts), or a changed ``asset_type``. The
    arguments are the ones the harness spy recorded; the expectation binds (ticker, ISO date)
    to this checkout's own ``propagate`` signature with its defaults."""
    res = ap_run(ACCT_JSON).assert_ok()
    graph_cls = _fork_symbol("TradingAgentsGraph", "tradingagents.graph.trading_graph",
                             "tradingagents.graph")
    bound = inspect.signature(graph_cls.propagate).bind(object(), TICKER, ISO_TRADE_DATE)
    bound.apply_defaults()
    expected = {k: json_safe(v) for k, v in list(bound.arguments.items())[1:]}
    assert len(res.propagate_calls) == 1, res.propagate_calls
    assert res.propagate_calls[0]["arguments"] == expected
