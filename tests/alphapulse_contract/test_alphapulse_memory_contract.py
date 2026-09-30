"""Memory log & settlement: the fork's feedback loop that alpha-pulse's runs rely on.

alpha-pulse gives every ``main.py`` run its own log —
``TRADINGAGENTS_MEMORY_LOG_PATH=<memory dir>/trading_memory_<TICKER>.md``, set for every
run it starts — and forces
``TRADINGAGENTS_CHECKPOINT_ENABLED=false``, so 4-8 parallel runs never share a file
(the fork rewrites a log through a fixed ``.tmp`` sibling). Inside each run:

1. the ticker's pending decisions are settled before the Portfolio Manager runs:
   5 trading days, alpha against the listing market's index (``^KS11`` for .KS,
   ``^KQ11`` for .KQ, ``SPY`` for US), one reflector call per entry;
2. the PM receives the settled lessons (same-day nightly runs: every lesson,
   including legacy tags; historical runs: only lessons known by the trade date);
3. the run appends its own decision as a ``pending`` entry tagged with the typed
   rating, account figures scrubbed (the saved report keeps them).

The logs on the server were written by a2981a7, so a merged reader must read them
exactly as a2981a7 does (``fixtures/memory_log_a2981a7_*``), and whatever the merged
writer appends must stay in the shape a2981a7 reads (rollback).

Prices come from the harness's yfinance library fake; the lesson is the scenario's
scripted reflector reply. Nothing in the fork is patched.
"""

from __future__ import annotations

import concurrent.futures
import datetime as dt
import shutil
from pathlib import Path

import pytest

from . import scenarios
from ._compat import ExpectedDrift
from ._memory_helpers import (
    AUDIT_OUT_ENV,
    AUDIT_WATCH_ENV,
    BENCHMARKS,
    LOG_FIXTURES,
    MEMORY_LOG_HOMES,
    PENDING_TAG_RE,
    RESOLVED_TAG_RE,
    SEPARATOR,
    blocks_for,
    dir_listing,
    entry_text,
    expected_readings,
    file_state,
    first_seq,
    memory_log_class,
    pending_tag,
    read_audit,
    resolved_tag,
    seed_log,
    settlement_price_calls,
    sha256_file,
    split_entries,
    sqlite_files,
    tag_line,
    write_audit_site,
    written_paths,
)
from .harness import FIXTURES, REPO_ROOT, kst_today, memory_log_name, run_main

S1 = "s1_nightly_kr_holding_sell"
S2 = "s2_pm_prose_quotes_other_ratings"
S2U = "s2u_pm_prose_quotes_other_ratings_underweight"
S4 = "s4_us_not_held_discovery"
S5A = "s5a_nightly_yyyymmdd_not_held"
S5B = "s5b_no_date_kst_today"

REIT = "417310.KS"
SAMSUNG = "005930.KS"
KOSDAQ = "247540.KQ"

# The reflector's scripted reply (scenario input) — the lesson a settlement stores.
LESSON = scenarios.get(S1)["roles"]["reflector"]["text"]

# Account figures s1/s2 put into the PM's prose (fixtures/position_contexts.json:
# held_qty 1250, avg_price 10480.0, cash 421675870.0), as the model writes them.
ACCOUNT_FIGURES = ("1,250", "10,480", "421,675,870")


# ============================================================================ scenarios

def _pending_entry(date: str, ticker: str, rating: str, decision: str) -> str:
    return entry_text(pending_tag(date, ticker, rating), decision)


KQ_INFO = {
    "symbol": KOSDAQ, "shortName": "에코프로비엠", "longName": "에코프로비엠",
    "quoteType": "EQUITY", "exchange": "KOE", "fullExchangeName": "KOSDAQ",
    "market": "kr_market", "currency": "KRW", "financialCurrency": "KRW",
    "country": "South Korea", "sector": "Industrials",
    "industry": "Electrical Equipment & Parts", "exchangeTimezoneName": "Asia/Seoul",
    "exchangeTimezoneShortName": "KST", "regularMarketPrice": 100.0, "currentPrice": 100.0,
    "trailingPegRatio": None,
}
KQ_PENDING_DECISION = ("**Rating**: Hold\n\n**Executive Summary**: 양극재 수요 둔화로 신규 매수는 "
                       "보류하고 기존 비중을 유지한다.\n\n**Investment Thesis**: 출하량 회복이 "
                       "확인되기 전까지는 방향성 베팅을 미룬다.")
US_PENDING_DECISION = ("**Rating**: Buy\n\n**Executive Summary**: Build to 3% on the services "
                       "margin inflection.\n\n**Investment Thesis**: Services revenue hit a record "
                       "and the device cycle is turning.")
SAMSUNG_PENDING_DECISION = ("**Rating**: Buy\n\n**Executive Summary**: HBM3E 공급 확대를 근거로 "
                            "2% 비중까지 두 번에 나눠 진입한다.\n\n**Investment Thesis**: 메모리 "
                            "업황 반등과 외국인 순매수 전환이 확인됐다.")


def _s1() -> dict:
    return scenarios.get(S1)


def _kq_past_date() -> dict:
    """A KOSDAQ listing (not in instruments.json: served via info_overrides) with one
    pending decision from a week before the historical trade date."""
    seed = seed_log(_pending_entry("2026-08-12", KOSDAQ, "Hold", KQ_PENDING_DECISION))
    return scenarios.derive(scenarios.get(S5B), {
        "argv": [KOSDAQ, "2026-08-19"],
        "memory_log_seed": seed,
        "yfinance": {"info_overrides": {KOSDAQ: KQ_INFO}},
    }, name="mem_settle_kq_benchmark", company="에코프로비엠")


def _us_past_date() -> dict:
    seed = seed_log(_pending_entry("2026-08-12", "AAPL", "Buy", US_PENDING_DECISION))
    return scenarios.derive(scenarios.get(S4), {"memory_log_seed": seed},
                            name="mem_settle_us_benchmark")


LEGACY_LESSON = "레거시 교훈: 분할 진입이 변동성을 흡수했다. 다음에는 첫 트랜치를 더 작게 가져간다."
DATED_LESSON = "해결일 교훈: 가이던스 상향은 이미 가격에 반영돼 있었다. 컨센서스 괴리를 먼저 본다."


def _live_kr(today: dt.date) -> tuple[dict, dict[str, str]]:
    """Same-day nightly run (no date argument, as_of=None) with a legacy 6-field lesson,
    a lesson with a resolution date, and a pending decision old enough to settle."""
    def ago(n: int) -> str:
        return (today - dt.timedelta(days=n)).isoformat()

    dates = {"legacy": ago(40), "dated": ago(30), "dated_resolved": ago(23), "pending": ago(14)}
    seed = seed_log(
        entry_text(resolved_tag(dates["legacy"], SAMSUNG, "Buy", "+2.1%", "+0.7%"),
                   "**Rating**: Buy\n\n**Executive Summary**: 3% 비중까지 두 번에 나눠 진입한다.",
                   LEGACY_LESSON),
        entry_text(resolved_tag(dates["dated"], SAMSUNG, "Overweight", "-1.4%", "+0.3%",
                                resolved=dates["dated_resolved"]),
                   "**Rating**: Overweight\n\n**Executive Summary**: 비중을 4%까지 늘린다.",
                   DATED_LESSON),
        _pending_entry(dates["pending"], SAMSUNG, "Buy", SAMSUNG_PENDING_DECISION),
    )
    spec = scenarios.derive(scenarios.get(S5B), {"memory_log_seed": seed},
                            name="mem_live_kr_lessons")
    return spec, dates


def _pm_first_seq(res) -> int:
    return first_seq(res.calls_for("portfolio_manager"))


def _assert_in_order(text: str, *needles: str) -> None:
    """Every needle occurs, each after the previous one."""
    pos = -1
    for needle in needles:
        found = text.find(needle, pos + 1)
        assert found > pos, f"{needle!r} missing or out of order after offset {pos}:\n{text[:4000]}"
        pos = found


# ============================================================================ reading the a2981a7 logs

def test_memory_log_class_under_test_is_the_one_the_graph_instantiates(ap_run):
    """Breaks if: a merge leaves an orphaned copy of the log class (e.g. a2981a7's
    agents/utils/memory.py beside upstream's decision_log.py / memory package) while the
    graph instantiates another one -- renamed or reached through a module attribute --, or
    the pytest process imports another checkout (the developer .venv's editable install
    points at the main checkout) — the reader tests below would then vouch for code
    production never runs. The class the graph instantiates is read from the live graph by
    the harness, not looked up by name."""
    cls = memory_log_class()  # asserts the defining file is under this checkout
    seen = ap_run(S1).assert_ok().propagate_calls[0].get("memory_log_classes")
    assert seen and len(seen) == 1, f"memory-log instances on the graph: {seen!r}"
    rec = seen[0]
    assert (rec["module"], rec["qualname"]) == (cls.__module__, cls.__qualname__), (
        f"the graph instantiates {rec['module']}.{rec['qualname']} but the reader tests "
        f"exercise {cls.__module__}.{cls.__qualname__} (candidates {MEMORY_LOG_HOMES})")
    assert rec["file"] and Path(rec["file"]).resolve().is_relative_to(REPO_ROOT.resolve()), rec


@pytest.mark.parametrize("name", LOG_FIXTURES)
def test_log_written_by_a2981a7_parses_to_the_reviewed_entries_and_pending_set(tmp_path, name):
    """Breaks if: a merged reader stops reading the logs already on the server — the
    legacy 6-field tag (no resolved: field) or the resolved:YYYY-MM-DD field no longer
    parses, a pending tag is not recognised (it would never be settled), the
    <!-- ENTRY_END --> delimiter changes — or reading migrates/rewrites the file
    (sha256, size, mtime or directory listing change)."""
    expected = expected_readings()["logs"][name]
    assert sha256_file(FIXTURES / name) == expected["sha256"], (
        f"fixture {name} changed; its reviewed reading in memory_log_a2981a7_expected.json "
        "no longer applies")
    memdir = tmp_path / "memory"
    memdir.mkdir()
    path = memdir / name
    shutil.copyfile(FIXTURES / name, path)
    before, listing = file_state(path), dir_listing(memdir)

    log = memory_log_class()({"memory_log_path": str(path)})
    entries = log.load_entries()
    pending = log.get_pending_entries()
    for ticker in {e["ticker"] for e in entries}:
        log.get_past_context(ticker)
        log.get_past_context(ticker, as_of="2026-09-10")

    assert len(entries) == expected["entries"]
    assert [[e["date"], e["ticker"], e["rating"]] for e in pending] == expected["pending"]
    assert file_state(path) == before, "reading the log modified it"
    assert dir_listing(memdir) == listing, "reading the log created or removed files"


def _past_context_cases():
    cases = []
    for name, reading in expected_readings()["logs"].items():
        for ticker, modes in reading["past_context"].items():
            for mode in modes:
                short = name.removeprefix("memory_log_a2981a7_").removesuffix(".md")
                cases.append(pytest.param(name, ticker, mode, id=f"{short}-{ticker}-{mode}"))
    return cases


@pytest.mark.parametrize("name,ticker,mode", _past_context_cases())
def test_log_written_by_a2981a7_gives_the_pm_the_reviewed_lessons(tmp_path, name, ticker, mode):
    """Breaks if: the lessons a merged reader builds from the server's existing logs
    change — order (most recent first), the n_same=5 / n_cross=3 limits, the rendered
    tag (resolved: dropped), the DECISION/REFLECTION layout, cross-ticker formatting, or
    the as_of cutoff (historical runs keep only lessons resolved by the trade date, so
    the legacy tag drops out; same-day runs keep every settled lesson)."""
    expected = expected_readings()["logs"][name]["past_context"][ticker][mode]
    path = tmp_path / name
    shutil.copyfile(FIXTURES / name, path)
    before = file_state(path)
    log = memory_log_class()({"memory_log_path": str(path)})
    if mode == "live":
        got = log.get_past_context(ticker)
    else:
        assert mode.startswith("as_of="), mode
        got = log.get_past_context(ticker, as_of=mode[len("as_of="):])
    assert got.split("\n") == expected
    assert file_state(path) == before


# ============================================================================ settlement in a real run

def test_pending_decision_is_settled_before_the_pm_and_the_pm_reads_its_lesson(ap_run):
    """Breaks if: a merge drops the settlement from propagate (e.g. deletes the fork's
    _resolve_pending_entries call without wiring upstream's settle_pending), settles after
    the lessons were read (the PM misses the newest lesson), swallows a reflector failure
    so the entry stays pending for good (upstream settle_pending + the fork's Reflector
    signature), fails on a stale import on the settlement path (rc 1 for every ticker
    with a pending entry), or stops handing the lessons to the Portfolio Manager."""
    res = ap_run(S1).assert_ok()
    seed_blocks = split_entries(res.memory_log_seed)
    old_block, pending_block = seed_blocks
    assert PENDING_TAG_RE.fullmatch(tag_line(pending_block)), pending_block

    # The earlier lesson is carried over byte for byte; the pending entry is settled in
    # place: resolved tag (5 trading days after 2026-08-12 on a Mon-Fri calendar is
    # 2026-08-19) + the unchanged decision + the reflector's lesson.
    blocks = split_entries(res.memory_log)
    assert blocks[0] == old_block
    settled = blocks_for(res.memory_log, "2026-08-12", REIT)
    assert len(settled) == 1, settled
    tag, body = settled[0].split("\n", 1)
    m = RESOLVED_TAG_RE.fullmatch(tag)
    assert m, tag
    assert (m["rating"], m["holding"], m["resolved"]) == ("Overweight", "5d", "2026-08-19"), tag
    assert body == pending_block.split("\n", 1)[1] + "\n\nREFLECTION:\n" + LESSON

    # One reflection, on the pending decision, before the Portfolio Manager's call.
    reflections = res.calls_for("reflector")
    assert len(reflections) == 1, [c.get("role") for c in res.llm_calls][:5]
    pending_decision = pending_block.split("DECISION:\n", 1)[1]
    assert pending_decision in res.prompt_text(reflections[0])
    assert int(reflections[0]["seq"]) < _pm_first_seq(res)

    # The PM reads both lessons, newest first.
    old_lesson = old_block.split("REFLECTION:\n", 1)[1]
    _assert_in_order(res.prompt_for("portfolio_manager"),
                     f"Past analyses of {REIT} (most recent first):",
                     f"[2026-08-12 | {REIT} | Overweight |", LESSON,
                     f"[2026-07-29 | {REIT} | Buy |", old_lesson)


BENCHMARK_CASES = [
    pytest.param(_s1, REIT, "^KS11", id="KOSPI-listing-vs-^KS11"),
    pytest.param(_kq_past_date, KOSDAQ, "^KQ11", id="KOSDAQ-listing-vs-^KQ11"),
    pytest.param(_us_past_date, "AAPL", "SPY", id="US-listing-vs-SPY"),
]


# (not named "benchmark": pytest-benchmark, installed in the developer .venv, owns that name)
@pytest.mark.parametrize("make_spec,ticker,index_symbol", BENCHMARK_CASES)
def test_settlement_measures_alpha_against_the_listing_markets_index(ap_run, make_spec, ticker,
                                                                    index_symbol):
    """Breaks if: the benchmark map loses .KS/.KQ (upstream v0.5.1's default_config has
    no Korean entries), so a Korean decision's alpha — and the lesson built on it — is
    measured against SPY; or an explicit benchmark override starts applying to every
    ticker. Observed at the library boundary: whose daily prices the settlement fetched
    for the pending entry's window before the PM ran, and what the reflector was told."""
    res = ap_run(make_spec()).assert_ok()
    fetched = settlement_price_calls(res, "2026-08-12", _pm_first_seq(res))
    assert {d["symbol"] for d in fetched} == {ticker, index_symbol}, fetched
    reflections = res.calls_for("reflector")
    assert len(reflections) == 1, len(reflections)
    prompt = res.prompt_text(reflections[0])
    assert index_symbol in prompt, prompt
    assert not [b for b in BENCHMARKS if b != index_symbol and b in prompt], prompt
    settled = blocks_for(res.memory_log, "2026-08-12", ticker)
    assert len(settled) == 1 and RESOLVED_TAG_RE.fullmatch(tag_line(settled[0])), settled


def test_same_day_run_gives_the_pm_every_settled_lesson_including_legacy_tags(ap_run):
    """Breaks if: a same-day run (alpha-pulse's nightly batch and web runs) starts
    filtering lessons by resolution date like a historical run — the lessons settled
    before #1251 (legacy 6-field tags, no resolved: field) would silently vanish from the
    Portfolio Manager's prompt — or stops settling in the live path."""
    before = kst_today()
    spec, dates = _live_kr(before)
    res = ap_run(spec).assert_ok()
    after = kst_today()
    assert res.trade_date_seen in {before.isoformat(), after.isoformat()}, res.trade_date_seen

    settled = blocks_for(res.memory_log, dates["pending"], SAMSUNG)
    assert len(settled) == 1, res.memory_log
    m = RESOLVED_TAG_RE.fullmatch(tag_line(settled[0]))
    assert m and m["resolved"] and m["resolved"] <= res.trade_date_seen, tag_line(settled[0])
    assert settled[0].endswith("REFLECTION:\n" + LESSON)

    _assert_in_order(res.prompt_for("portfolio_manager"),
                     f"Past analyses of {SAMSUNG} (most recent first):",
                     f"[{dates['pending']} | {SAMSUNG} | Buy |", LESSON,
                     f"[{dates['dated']} | {SAMSUNG} | Overweight |", DATED_LESSON,
                     f"[{dates['legacy']} | {SAMSUNG} | Buy |", LEGACY_LESSON)


# ============================================================================ what a run appends

@pytest.mark.parametrize("name,typed_rating", [(S1, "Sell"), (S2, "Sell"), (S2U, "Underweight")])
def test_run_appends_one_pending_entry_tagged_with_the_typed_rating(ap_run, name, typed_rating):
    """Breaks if: the memory tag stops following the PM's typed rating — upstream v0.5.1's
    'last label wins' parser reads 'consensus rating: Hold' out of the kill-switch prose
    (s2/s2u), so the next run's lessons and settlement report a call nobody made; the
    record path loses scrub_account_numbers (upstream record_decision has none), so raw
    balances re-enter the next five PM prompts; the tag stops being the ISO
    '[YYYY-MM-DD | TICKER | Rating | pending]' a2981a7 reads back after a rollback; or
    the entry is appended twice / not delimited."""
    res = ap_run(name).assert_ok()
    text = res.memory_log
    assert text.endswith(SEPARATOR), text[-200:]
    blocks = split_entries(text)
    assert len(blocks) == len(split_entries(res.memory_log_seed)) + 1
    new = blocks[-1]
    assert tag_line(new) == f"[2026-08-19 | {REIT} | {typed_rating} | pending]"
    assert new.split("\n", 1)[1].startswith(f"\nDECISION:\n**Rating**: {typed_rating}\n")
    assert len(blocks_for(text, "2026-08-19", REIT)) == 1
    # Account figures are scrubbed from the archived copy only; the report keeps them.
    assert "[redacted]" in new
    leaked = [f for f in ACCOUNT_FIGURES if f in text]
    assert leaked == [], leaked
    assert res.complete_report is not None
    assert [f for f in ACCOUNT_FIGURES if f not in res.complete_report] == []


def test_run_reads_and_writes_only_the_per_ticker_log_named_by_the_env(ap_run):
    """Breaks if: TRADINGAGENTS_MEMORY_LOG_PATH stops being honoured (a merged
    default_config/main.py falls back to ~/.tradingagents/memory/trading_memory.md or
    another location) — every parallel run would share one file again, and the per-ticker
    history alpha-pulse keeps would no longer be read or written."""
    res = ap_run(S1).assert_ok()
    expected = Path(res.env["TRADINGAGENTS_MEMORY_LOG_PATH"])
    assert expected.name == memory_log_name(REIT)
    # The seeded history was read from that file (its pending entry got settled there)
    # and this run's decision was appended to it.
    assert res.memory_log is not None
    assert split_entries(res.memory_log)[0] == split_entries(res.memory_log_seed)[0]
    assert RESOLVED_TAG_RE.fullmatch(tag_line(blocks_for(res.memory_log, "2026-08-12", REIT)[0]))
    assert len(blocks_for(res.memory_log, "2026-08-19", REIT)) == 1
    others = sorted(str(p.relative_to(res.run_dir)) for p in res.run_dir.rglob("*trading_memory*")
                    if p.is_file() and p.resolve() != expected.resolve())
    assert others == [], others
    assert not (res.home / ".tradingagents").exists()


def test_parallel_runs_with_per_ticker_logs_never_touch_each_others_files(tmp_path):
    """Breaks if: the atomic rewrite of a log uses a temp name that is not unique to that
    log (a fixed 'trading_memory.tmp', a name derived from the directory, ...), or one
    run's settlement touches another ticker's log. alpha-pulse runs 4-8 tickers at once
    with every per-ticker log in ONE directory and shares HOME/TMPDIR/results/cache, so a
    shared temp file means a crashed run (FileNotFoundError on replace) or one ticker's
    history written over another's. Deterministic: an audit hook records every file each
    run wrote under the memory directory; the two sets must not overlap."""
    shared = (tmp_path / "shared").resolve()
    memdir = shared / "apmem_watch_memory"
    for sub in (memdir, shared / "home", shared / "tmp", shared / "results", shared / "cache"):
        sub.mkdir(parents=True)
    audit_site = write_audit_site(tmp_path / "audit_site")
    common = {
        "HOME": str(shared / "home"), "TMPDIR": str(shared / "tmp"),
        "TRADINGAGENTS_RESULTS_DIR": str(shared / "results"),
        "TRADINGAGENTS_CACHE_DIR": str(shared / "cache"),
        "PYTHONPATH": str(audit_site), AUDIT_WATCH_ENV: str(memdir),
    }
    samsung_seed = seed_log(_pending_entry("2026-08-12", SAMSUNG, "Buy", SAMSUNG_PENDING_DECISION))
    runs = {
        REIT: scenarios.derive(scenarios.get(S1), {}, name="mem_parallel_417310"),
        SAMSUNG: scenarios.derive(scenarios.get(S5A), {"memory_log_seed": samsung_seed},
                                  name="mem_parallel_005930"),
    }
    logs = {t: memdir / memory_log_name(t) for t in runs}
    audits = {t: tmp_path / f"audit_{t}.jsonl" for t in runs}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = {t: pool.submit(run_main, tmp_path / "runs", spec, None,
                                  {**common, "TRADINGAGENTS_MEMORY_LOG_PATH": str(logs[t]),
                                   AUDIT_OUT_ENV: str(audits[t])})
                   for t, spec in runs.items()}
        results = {t: f.result() for t, f in futures.items()}
    for res in results.values():
        res.assert_ok()

    touched = {t: written_paths(read_audit(audits[t])) for t in runs}
    for t, other in ((REIT, SAMSUNG), (SAMSUNG, REIT)):
        # Non-vacuous: the observer saw this run write its own log.
        assert str(logs[t]) in touched[t], (t, sorted(touched[t]))
        assert str(logs[other]) not in touched[t], (t, sorted(touched[t]))
    scratch = {t: touched[t] - {str(logs[t])} for t in runs}
    assert scratch[REIT].isdisjoint(scratch[SAMSUNG]), sorted(scratch[REIT] & scratch[SAMSUNG])
    assert dir_listing(memdir) == sorted(p.name for p in logs.values()), dir_listing(memdir)

    for t, path in logs.items():
        text = path.read_text(encoding="utf-8")
        assert {tag_line(b).split(" | ")[1] for b in split_entries(text)} == {t}, text
        settled = blocks_for(text, "2026-08-12", t)
        assert len(settled) == 1 and RESOLVED_TAG_RE.fullmatch(tag_line(settled[0])), settled
        new = blocks_for(text, "2026-08-19", t)
        assert len(new) == 1 and PENDING_TAG_RE.fullmatch(tag_line(new[0])), new
    assert not (shared / "home" / ".tradingagents").exists()


# ============================================================================ checkpoints

def test_checkpoint_disabled_by_env_leaves_no_checkpoint_database(ap_run):
    """Breaks if: TRADINGAGENTS_CHECKPOINT_ENABLED=false stops turning checkpointing off
    (the env mapping is dropped while a default turns it on, or 'false' is read as a
    truthy string) — parallel same-ticker runs would share a SQLite checkpoint, and a
    resumed run would skip the position context it was not keyed on."""
    res = ap_run(S1).assert_ok()
    assert res.env["TRADINGAGENTS_CHECKPOINT_ENABLED"] == "false"
    dbs = sqlite_files(res.run_dir)
    config = res.graph_config or {}
    assert dbs == [], (dbs, config.get("checkpoint_enabled"))
    assert not (res.cache_dir / "checkpoints").exists()


def test_checkpoint_env_variable_is_still_the_switch(ap_run):
    """Breaks if: the fork stops reading TRADINGAGENTS_CHECKPOINT_ENABLED at all (renamed
    or removed env mapping) — alpha-pulse's explicit 'false' would then guard nothing the
    day a default turns checkpoints on, and the test above could pass vacuously. Control:
    with 'true' the same run must leave a SQLite checkpoint under the run's directories."""
    spec = scenarios.derive(scenarios.get(S1), {"memory_log_seed": None},
                            name="mem_checkpoint_on_control")
    res = ap_run(spec, env_overrides={"TRADINGAGENTS_CHECKPOINT_ENABLED": "true"}).assert_ok()
    assert sqlite_files(res.run_dir), sorted(
        str(p.relative_to(res.run_dir)) for p in res.run_dir.rglob("*") if p.is_file())[:40]


# ============================================================================ expected drift

def test_rerun_on_an_already_settled_date_does_not_record_the_decision_twice(ap_run):
    """Expected drift, not a contract today. Once the merge makes it pass, it catches: a
    re-run (web re-analysis of a settled date) appending a second entry for the same
    ticker and date, which the next runs would count twice in their lessons."""
    earlier_lesson = "보유 유지 교훈: 공시 일정이 가격을 움직였다. 다음에는 일정부터 확인한다."
    seed = seed_log(
        entry_text(resolved_tag("2026-08-12", REIT, "Overweight", "+2.5%", "-0.0%",
                                resolved="2026-08-19"),
                   "**Rating**: Overweight\n\n**Executive Summary**: 공시 전까지 보유를 유지한다.",
                   earlier_lesson),
        entry_text(resolved_tag("2026-08-19", REIT, "Hold", "+0.4%", "+0.1%", resolved="2026-08-26"),
                   "**Rating**: Hold\n\n**Executive Summary**: 추가 매수 없이 관망한다.",
                   "관망은 결과적으로 중립이었다. 다음에는 금리 일정을 먼저 본다."),
    )
    spec = scenarios.derive(scenarios.get(S1), {"memory_log_seed": seed},
                            name="mem_rerun_settled_date")
    # Preconditions are real failures, never the drift: the run completed and read THIS
    # per-ticker log (the lesson resolved by the trade date reached the PM), so an
    # unchanged log below means "not re-recorded", not "never looked at".
    res = ap_run(spec).assert_ok()
    assert earlier_lesson in res.prompt_for("portfolio_manager")
    same_day = blocks_for(res.memory_log, "2026-08-19", REIT)
    if len(same_day) != 1:
        raise ExpectedDrift(f"{len(same_day)} entries for 2026-08-19: "
                            f"{[tag_line(b) for b in same_day]}")
