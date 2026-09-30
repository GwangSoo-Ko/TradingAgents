"""Behaviour the upstream v0.5.x merge is EXPECTED to change — pinned as strict xfails.

The other contract tests freeze what alpha-pulse reads and must stay green through the
merge. The four behaviours here are different: the merge review accepted that they
change, and each change is visible to alpha-pulse (a run that used to fail succeeds, or
the other way round, or the per-ticker memory log gets different entries). So each test
asserts the NEW behaviour precisely, through the real main.py run the way alpha-pulse
runs it (``harness.run_main``: data faked at the yfinance/requests layer, the model at
the LLM boundary):

* on a2981a7 (production today) it sees the old behaviour and raises ``ExpectedDrift``
  -> XFAIL;
* once the merge brings the upstream change it passes -> strict XPASS fails the suite,
  which tells whoever merges to delete the marker; from then on the test is a contract;
* anything else — a broken precondition, or a merge that lands only half of the change
  — is a plain AssertionError and fails for real (``raises=ExpectedDrift``).

Upstream references: d04693a (future trade date), ef383df (unsettled newest bar),
tradingagents/graph/settlement.py and tradingagents/decision_log.py @ 35543d0 (v0.5.1;
the same code lives in tradingagents/memory/ in v0.5.2).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from typing import Any

import pytest

from . import scenarios
from ._compat import ExpectedDrift
from ._fake_data import close_on
from ._memory_helpers import (
    RESOLVED_TAG_RE,
    blocks_for,
    entry_text,
    pending_tag,
    seed_log,
    split_entries,
    tag_line,
)
from ._merge_expectations_helpers import (
    FAULT_KEY,
    FAULT_MARKER,
    answered_calls,
    assert_ran_this_checkout,
    chat_fault,
    exit_info,
    last_weekday_on_or_before,
    previous_weekday,
    raised_calls,
    snapshot_reading,
    write_llm_fault_site,
)
from .harness import RunResult, kst_today, run_main

S1 = "s1_nightly_kr_holding_sell"
S5B = "s5b_no_date_kst_today"
REIT = "417310.KS"      # s1: the KR holding (하나글로벌리츠), nightly date 20260819
SAMSUNG = "005930.KS"   # s5b: KR candidate, no date argument (KST today)
TRADE_DATE = "2026-08-19"

# The reflector's scripted reply (scenario input): the lesson a settlement stores.
LESSON = scenarios.get(S1)["roles"]["reflector"]["text"]


def expected_drift(what: str) -> pytest.MarkDecorator:
    return pytest.mark.xfail(
        strict=True, raises=ExpectedDrift,
        reason=f"expected to change at the upstream v0.5.x merge: {what} "
               "Delete this marker in the merge commit once the test passes.")


def _within_one_kst_day(attempt: Callable[[], Any]) -> Any:
    """Run ``attempt`` so that it does not straddle midnight KST.

    Two tests derive their expectation from 'today in KST' (the subprocess runs with
    TZ=Asia/Seoul). If the date turned while a run was in flight, run once more; two
    back-to-back runs of a few seconds cannot both straddle a midnight.
    """
    for _ in range(2):
        before = kst_today()
        outcome = attempt()
        if kst_today() == before:
            return outcome
    raise AssertionError("two consecutive runs straddled midnight KST")


def _pm_first_seq(res: RunResult) -> int:
    calls = res.calls_for("portfolio_manager")
    assert calls, f"the Portfolio Manager never ran\n{res.describe()}"
    return min(int(c["seq"]) for c in calls)


# ============================================================================ (1) future date

@expected_drift(
    "propagate() rejects a trade date after today (KST) with ValueError before it settles, "
    "fetches or calls a model (upstream d04693a, #1319); a2981a7 runs the whole analysis.")
def test_future_trade_date_fails_the_run_before_any_model_call_or_write(tmp_path):
    """Expected drift, not a contract today: a2981a7 analyses a date that has not traded.

    alpha-pulse's web entry accepts any valid ISO date (no upper bound), so KST tomorrow
    reaches ``main.py TICKER YYYY-MM-DD``. Once the merge makes this
    pass (marker removed), it catches: a merge that drops or weakens upstream's
    trade-date guard, or runs it after the settlement, the data layer or a model call —
    a future-date web request would again buy a full Opus/Sonnet run (and settle the
    ticker's pending entries) for a day that has not traded, and append a pending memory
    entry for it.

    New behaviour, as alpha-pulse sees it: main.py exits non-zero (run 'failed': no
    decision, no plan, no report) because propagate() raised a ValueError naming the
    future ISO date, and nothing was spent or written first: zero model calls, zero
    data calls, the per-ticker memory log byte-identical (its due pending entry is not
    settled). The LLM clients are still built with the graph before propagate(); a
    construction sends no request, so the zero that matters is model calls.
    """
    def attempt() -> tuple[RunResult, dt.date]:
        tomorrow = kst_today() + dt.timedelta(days=1)
        spec = scenarios.derive(scenarios.get(S1), {}, name="mx_future_trade_date")
        return run_main(tmp_path, spec, argv=[REIT, tomorrow.isoformat()]), tomorrow

    res, tomorrow = _within_one_kst_day(attempt)
    assert_ran_this_checkout(res)
    # Precondition (a2981a7 and merged): main.py accepted the ISO date and handed it to
    # propagate unchanged, so whatever happens next happens inside propagate.
    assert res.trade_date_seen == tomorrow.isoformat(), res.describe()
    seed = res.memory_log_seed
    assert seed and "| pending]" in seed, "s1 must seed a pending entry that is due"

    # a2981a7's signature: the whole analysis ran and printed a decision. (rc 0 without
    # an analysis — e.g. main.py swallowing the error — is not the drift: it fails below.)
    if res.rc == 0 and res.llm_calls and res.decision is not None:
        raise ExpectedDrift(
            f"a2981a7 analysed {tomorrow} although today is {tomorrow - dt.timedelta(days=1)} "
            f"in KST: rc 0, {len(res.llm_calls)} model calls, decision {res.decision!r}, "
            f"memory log {'rewritten' if res.memory_log != seed else 'unchanged'}")

    exc = exit_info(res)
    assert (exc.get("kind"), exc.get("type")) == ("exception", "ValueError"), res.describe()
    message = str(exc.get("message") or "")
    assert tomorrow.isoformat() in message and "future" in message.lower(), message
    assert res.llm_calls == [], [(c.get("role"), c.get("kind")) for c in res.llm_calls]
    assert res.data_calls == [], [d.get("api") or d.get("url") for d in res.data_calls]
    assert res.memory_log == seed, "the memory log changed although the date was rejected"
    assert (res.trade_plan_lines, res.report_path) == ([], None), res.tail
    assert not (res.results_dir / "reports").exists()
    # alpha-pulse marks a non-zero exit 'failed' and keeps neither decision nor plan.
    assert res.rc != 0, res.describe()


# ============================================================================ (2) NaN newest close

UNSETTLED = {"api": "download", "symbol": SAMSUNG, "name": "unsettled-newest-bar",
             "nan_last_close": True}


@expected_drift(
    "a newest bar without a close is an unsettled session: the verified snapshot serves the "
    "last settled bar (upstream ef383df, #1289); a2981a7 raises NoMarketDataError from the "
    "snapshot tool and the run fails.")
def test_verified_snapshot_uses_the_last_settled_bar_when_the_newest_close_is_missing(tmp_path):
    """Expected drift, not a contract today: on a2981a7 a newest daily bar without a
    close (Yahoo's in-progress or glitched session) makes load_ohlcv raise; the market
    analyst's get_verified_market_snapshot re-raises through the ToolNode and main.py
    exits 1, discarding the run. Once the merge makes this pass (marker removed), it
    catches: a merge that keeps that raise (live runs over an unsettled session fail
    again — US web/discovery runs in the KST evening and early morning, KR runs before
    the open), and a fix that fabricates the missing close (forward-filling it would
    put the previous session's price under today's date in the snapshot the analyst
    must quote).

    The fault is injected at the library layer (the 5-year ``yf.download`` the fork
    caches returns its newest bar with Close = NaN); the run is a live KR run (no date
    argument -> KST today, alpha-pulse's fallback when it has no date), so the
    unsettled bar is the newest one on or before the trade date.

    New behaviour: rc 0 with the usual decision and plan, and the snapshot the market
    analyst receives names the last settled bar — the weekday before the unsettled one
    — as its latest row, with the close the library served for that day.
    """
    def attempt() -> tuple[RunResult, dt.date]:
        today = kst_today()
        spec = scenarios.derive(scenarios.get(S5B), {"yfinance": {"faults": [UNSETTLED]}},
                                name="mx_unsettled_newest_close")
        return run_main(tmp_path, spec), today

    res, today = _within_one_kst_day(attempt)
    assert_ran_this_checkout(res)
    # Preconditions (a2981a7 and merged): a live run, the fault really hit the price
    # download of the analysed ticker, and the analyst did ask for the snapshot.
    assert res.trade_date_seen == today.isoformat(), res.describe()
    faulted = [d for d in res.data_calls if d.get("api") == "download"
               and d.get("symbol") == SAMSUNG and d.get("fault") == UNSETTLED["name"]]
    assert faulted, [(d.get("api"), d.get("symbol"), d.get("fault")) for d in res.data_calls]
    requested = [tc for c in res.calls_for("market_analyst")
                 for tc in (c.get("reply") or {}).get("tool_calls") or []
                 if tc.get("name") == "get_verified_market_snapshot"]
    assert requested, "the market analyst never asked for the verified snapshot"

    exc = exit_info(res)
    if (res.rc != 0 and exc.get("type") == "NoMarketDataError"
            and "closing price" in str(exc.get("message"))):
        raise ExpectedDrift(f"a2981a7 failed the run over the unsettled bar: rc {res.rc}, "
                            f"{exc.get('type')}: {exc.get('message')}")

    res.assert_ok()
    # As alpha-pulse stores it: done (rc 0, the report line), decision Buy, a plan.
    assert (res.report_path is not None, res.decision) == (True, "Buy"), res.tail
    assert res.plan is not None, res.tail

    unsettled = last_weekday_on_or_before(today)   # newest bar served (Mon-Fri calendar)
    settled = previous_weekday(unsettled)          # the last bar with a close
    instrument = scenarios.fixture_json("instruments.json")[SAMSUNG]
    snapshots = [t["content"] for t in res.tool_results("market_analyst")
                 if t["name"] == "get_verified_market_snapshot"]
    assert len(snapshots) == 1, snapshots
    reading = snapshot_reading(str(snapshots[0]))
    assert reading["requested"] == today.isoformat(), reading
    assert reading["latest_row"] == settled.isoformat(), (reading, unsettled, snapshots[0][:1500])
    assert reading["close"] is not None, reading
    assert abs(reading["close"] - close_on(instrument, settled)) < 0.005, (
        reading, close_on(instrument, settled))


# ============================================================================ (3) reflection error

# Two due pending decisions (5 Mon-Fri trading days have passed for both by 2026-08-19).
# The provider fails the reflection of the first one only.
FAILING_MARK = "8월 5일 첫 진입"
FAILING_ENTRY = entry_text(
    pending_tag("2026-08-05", REIT, "Buy"),
    "**Rating**: Buy\n\n**Executive Summary**: 8월 5일 첫 진입 — 순자산 대비 할인 폭이 커 "
    "1% 비중으로 시작한다.\n\n**Investment Thesis**: 임대율이 안정적이고 배당수익률이 섹터 평균보다 높다.")
SETTLING_ENTRY = entry_text(
    pending_tag("2026-08-12", REIT, "Overweight"),
    "**Rating**: Overweight\n\n**Executive Summary**: 금리 결정 전까지 비중을 2%로 늘린다.\n\n"
    "**Investment Thesis**: 배당 매력이 차환 비용 부담보다 크다.")


@expected_drift(
    "a provider error in the settlement's reflection call leaves that entry pending and the "
    "run goes on (upstream settlement.settle_pending); a2981a7 lets it end the run (rc 1).")
def test_reflection_error_leaves_the_entry_pending_and_the_run_completes(tmp_path):
    """Expected drift, not a contract today: on a2981a7 the settlement at the start of
    propagate() calls the reflector unguarded, so one Vertex 429 on that Sonnet call
    ends the run with rc 1 before any analysis (web: a failed analysis; nightly: a
    failed ticker), again on every run of that ticker until the quota recovers.

    Once the merge makes this pass (marker removed), it catches: a merge that keeps
    the unguarded call; a fix that swallows the error for the whole settlement (the
    other due entry would stay unsettled and its lesson would not reach the PM); and a
    fix that marks the failed entry settled without a lesson (its outcome would never
    be reflected) or loses it.

    The provider error is injected at the LLM boundary (``_merge_expectations_helpers``
    shim: anthropic.RateLimitError, what ChatAnthropicVertex raises on HTTP 429) for
    the reflection of the 2026-08-05 entry only.

    New behaviour: rc 0 with the usual decision and plan; the 2026-08-05 entry stays
    pending byte for byte (the next run retries it); the 2026-08-12 entry is settled
    with its lesson (5 trading days -> resolved 2026-08-19) and the PM reads that
    lesson; this run's decision is appended as pending; the failure is reported on
    stderr, not swallowed silently.
    """
    site = write_llm_fault_site(tmp_path / "llm_fault_site")
    spec = scenarios.derive(scenarios.get(S1), {
        "memory_log_seed": seed_log(FAILING_ENTRY, SETTLING_ENTRY),
        "roles": {"reflector": {FAULT_KEY: chat_fault(prompt_contains=FAILING_MARK)}},
    }, name="mx_reflection_provider_error")
    res = run_main(tmp_path, spec, env_overrides={"PYTHONPATH": str(site)})
    assert_ran_this_checkout(res)
    # Precondition (a2981a7 and merged): the provider error really hit the reflection
    # of the 2026-08-05 entry (the shim was active and the entry was due).
    failed = [c for c in raised_calls(res, "reflector") if FAILING_MARK in res.prompt_text(c)]
    assert failed, [(c.get("role"), c.get("reply")) for c in res.calls_for("reflector")]
    raised = str(failed[0]["reply"]["raised"])
    assert FAULT_MARKER in raised, raised

    exc = exit_info(res)
    if (res.rc != 0 and exc.get("kind") == "exception"
            and raised.startswith(f"{exc.get('type')}:") and FAULT_MARKER in str(exc.get("message"))):
        raise ExpectedDrift(f"a2981a7 let the reflection error end the run: rc {res.rc}, "
                            f"{exc.get('type')}: {exc.get('message')}; "
                            f"{len(res.llm_calls)} model call(s), memory log "
                            f"{'unchanged' if res.memory_log == res.memory_log_seed else 'changed'}")

    res.assert_ok()
    # As alpha-pulse stores it: done (rc 0, the report line), decision Sell, a plan.
    assert (res.report_path is not None, res.decision) == (True, "Sell"), res.tail
    assert res.plan is not None, res.tail

    blocks = split_entries(res.memory_log)
    assert len(blocks) == 3, res.memory_log
    assert blocks[0] == FAILING_ENTRY, "the entry whose reflection failed must stay pending as is"

    tag, body = blocks[1].split("\n", 1)
    m = RESOLVED_TAG_RE.fullmatch(tag)
    assert m, tag
    assert (m["date"], m["rating"], m["holding"], m["resolved"]) == (
        "2026-08-12", "Overweight", "5d", TRADE_DATE), tag
    assert body == SETTLING_ENTRY.split("\n", 1)[1] + "\n\nREFLECTION:\n" + LESSON
    assert tag_line(blocks[2]) == pending_tag(TRADE_DATE, REIT, "Sell"), blocks[2][:200]

    answered = answered_calls(res, "reflector")
    assert len(answered) == 1 and "금리 결정 전까지 비중을 2%로 늘린다." in res.prompt_text(answered[0])
    pm_seq = _pm_first_seq(res)
    assert all(int(c["seq"]) < pm_seq for c in res.calls_for("reflector"))
    assert LESSON in res.prompt_for("portfolio_manager")
    assert FAULT_MARKER in res.stderr, "the failed reflection was swallowed without a trace"


# ============================================================================ (4) settled date

SAME_DAY_DECISION = ("**Rating**: Hold\n\n**Executive Summary**: 8월 19일 첫 분석 — 금리 결정 전까지 "
                     "관망한다.\n\n**Investment Thesis**: 금리 경로가 분명해지기 전에는 방향을 정하지 않는다.")
SAME_DAY_ENTRY = entry_text(pending_tag(TRADE_DATE, REIT, "Hold"), SAME_DAY_DECISION)


@expected_drift(
    "a ticker+date that already has an entry, pending or settled, is not recorded again "
    "(upstream decision_log.store_decision); a2981a7's guard only looks at pending tags, so "
    "a re-run that settles its own date appends that date a second time.")
def test_rerun_that_settles_its_own_date_does_not_record_that_date_again(ap_run, tmp_path):
    """Expected drift, not a contract today: a web re-analysis of a past date whose
    earlier entry is still pending (no run of that ticker settled it in between, e.g. a
    discovery name) first settles that entry, then a2981a7's record guard — looking
    only for a *pending* tag — appends a second entry for the same ticker and date,
    and the ticker's later lessons count that day twice.

    Once the merge makes this pass (marker removed), it catches: a merge that keeps
    a2981a7's pending-only guard, and an over-broad guard that stops recording dates
    that have no entry (control: s1 records 2026-08-19 although its log already
    mentions 'resolved:2026-08-19'). Complements test_alphapulse_memory_contract's
    xfail for a date that was settled before the run started.

    New behaviour: the settled earlier entry (Hold, 5 trading days -> resolved
    2026-08-26, with its lesson) is the only entry for 2026-08-19; this run's decision
    is not appended; stdout (decision, plan) is unaffected.
    """
    control = ap_run(S1).assert_ok()
    recorded = blocks_for(control.memory_log, TRADE_DATE, REIT)
    assert [tag_line(b) for b in recorded] == [pending_tag(TRADE_DATE, REIT, "Sell")], (
        control.memory_log)
    assert f"resolved:{TRADE_DATE}]" in control.memory_log, control.memory_log

    spec = scenarios.derive(scenarios.get(S1), {"memory_log_seed": seed_log(SAME_DAY_ENTRY)},
                            name="mx_rerun_settles_its_own_date")
    res = run_main(tmp_path, spec).assert_ok()
    assert_ran_this_checkout(res)
    # Preconditions (a2981a7 and merged): this run settled the earlier same-day entry.
    reflections = res.calls_for("reflector")
    assert len(reflections) == 1, len(reflections)
    assert SAME_DAY_DECISION in res.prompt_text(reflections[0])
    same_day = blocks_for(res.memory_log, TRADE_DATE, REIT)
    assert same_day, res.memory_log
    tag, body = same_day[0].split("\n", 1)
    m = RESOLVED_TAG_RE.fullmatch(tag)
    assert m, tag
    assert (m["rating"], m["holding"], m["resolved"]) == ("Hold", "5d", "2026-08-26"), tag
    assert body == SAME_DAY_ENTRY.split("\n", 1)[1] + "\n\nREFLECTION:\n" + LESSON

    # a2981a7's signature: this run's own decision appended right after the settled entry.
    if len(same_day) == 2 and tag_line(same_day[1]) == pending_tag(TRADE_DATE, REIT, "Sell"):
        raise ExpectedDrift(f"a2981a7 recorded {TRADE_DATE} again after settling it: "
                            f"{[tag_line(b) for b in same_day]}")

    assert split_entries(res.memory_log) == same_day, res.memory_log
    assert (res.decision, (res.plan or {}).get("rating")) == ("Sell", "Sell"), res.tail
