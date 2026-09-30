# alpha-pulse contract lock

Tests in this directory freeze **what the production consumer, alpha-pulse, reads
from this fork** — so an upstream merge that keeps every existing test green but
silently changes alpha-pulse's input fails here first.

They do not freeze internals. They freeze exactly what alpha-pulse consumes:

| Contract | What alpha-pulse does with it |
|---|---|
| `python main.py TICKER [DATE]` with DATE as ISO (web/discovery) or KST `YYYYMMDD` (nightly); no DATE ⇒ the process's own date (TZ, `Asia/Seoul` in production), rc 0 required | rc != 0 ⇒ the run failed and its plan is discarded |
| stdout grammar: the decision line (`buy/overweight/hold/underweight/sell/review`) right before the LAST `Report saved: <dir>/complete_report.md` line; a decision with no readable rating reads `REVIEW`, never a tradeable word | the run's decision on its dashboard, in its notifications and in the discovery report; REVIEW means "look at it", never a trade |
| the LAST `TRADE_PLAN_JSON: {...}` line: field names, vocabularies, all-or-nothing numbers (unreadable numbers ⇒ no plan, never a null-patched plan) | order drafts (its trade-plan module, named in main.py's comment); no line ⇒ "no plan" (`plan_status='unparsed'`) |
| report tree (`complete_report.md` + section files) | web report view, discovery deep report |
| env semantics: `TRADINGAGENTS_{RESULTS_DIR,CACHE_DIR,MEMORY_LOG_PATH,CHECKPOINT_ENABLED,POSITION_CONTEXT}` (`''` = not injected) | set for every run: per-run paths, a per-ticker memory log, checkpoints off, the account JSON or `''` |
| account context reaches the Portfolio Manager only (holdings + Founding Thesis + revision instruction); account numbers scrubbed from the archived memory-log copy only | its revision gate assumes the PM saw the thesis it sent |
| per-role LLM: RM/PM `claude-opus-5` effort max, all other roles `claude-sonnet-5` effort high, `max_tokens` 20000, thinking adaptive, Vertex (`GOOGLE_CLOUD_PROJECT`, location `global`), Korean output, KR vendors | production runs `main.build_config()` unmodified |

Contract docs on the fork side: `docs/INTEGRATION.md` (§1b the output and the plan
contract, §4 `TRADINGAGENTS_POSITION_CONTEXT`) and main.py's comments.

## Public tests here, consumer replays elsewhere

This directory is public. It holds the harness and the contract tests, and nothing of
the consumer:

* stdout is read with **`contract_reader.py`**, a clean-room reader of the output
  grammar the fork documents itself (docs/INTEGRATION.md §1b, main.py's comments at the
  `TRADE_PLAN_JSON` print, the `REVIEW` sentinel of `tradingagents/agents/utils/rating.py`)
  — lines as `print()` ends them (at `"\n"` only: a plan whose prose holds U+2028 is still
  one line), the LAST `Report saved: <path>` line, the decision word before it, the LAST
  `TRADE_PLAN_JSON: {...}` line as one strict JSON object, `is_review`. The sanity tests
  pin that grammar with hand-written stdout;
* every plan fixture is **synthetic** (`fixtures/synthetic_*.json`: hand-written,
  fictional numbers, neutral text) and so are the account JSONs of
  `fixtures/position_contexts.json`, the founding thesis in them, the memory-log seed and
  the news, posts and consensus the data fakes serve (`fixtures/instruments.json`,
  `fixtures/http_routes.json`): made-up, sector-level stories, no recorded plan or event;
* assertions are about what the fork emits — rc, the decision line (REVIEW included),
  the presence or absence of the plan line, plan fields and values, the report tree, env
  semantics, prompts, the memory log, LLM kwargs — never about the consumer's own
  post-processing.

alpha-pulse's own parser, validators and position-context builder are replayed against
these same runs (and its recorded production plans) by a **private companion suite that
lives outside this repository**. It imports `harness`, `scenarios`, `contract_reader`
and some test-module constants from here: when you rename one of those, expect the
companion suite to need the same change.

## How a test sees a run

`harness.run_main(tmp_path, scenario)` runs **the real `main.py`** in a subprocess the
way alpha-pulse does — `python tests/alphapulse_contract/_bootstrap.py <run_dir>` with
`sys.path[0]` = repo root, cwd = repo root, `runpy.run_path(main.py, run_name="__main__")`
— and fakes only external boundaries:

* **LLM** — a scripted langchain `BaseChatModel` (`_fake_llm.py`) installed at
  `create_llm_client` (`llm_boundary: "factory"`, default) or at the Vertex SDK class
  `ChatAnthropicVertex` (`llm_boundary: "sdk"`, so the fork's real kwargs translation
  runs). Structured output mirrors `ChatAnthropicVertex.with_structured_output`: the raw
  dict is validated by the fork's real pydantic schema, so real validators/coercers run
  and a ValidationError drives the fork's real free-text fallback.
* **Data** — at the library layer (`_fake_data.py`): `yfinance.Ticker/download/Search`
  serve `fixtures/instruments.json` plus a deterministic price path; `requests`
  (`HTTPAdapter.send`) and `urllib` (`OpenerDirector.open`) serve
  `fixtures/http_routes.json`. The fork's vendor code, router, caches and validators run.
* **Network** — sockets and `curl_cffi` (which bypasses Python sockets) are refused and
  recorded; unknown URLs get a 404 and are recorded as `unrouted`.
* **Environment** — built from scratch (nothing inherited from your shell): HOME,
  TMPDIR, results/cache/memory under the run dir, `TZ=Asia/Seoul`,
  `GOOGLE_CLOUD_PROJECT=tpmn-dev`, `PYTHONUNBUFFERED=1`, `PYTHONDONTWRITEBYTECODE=1`,
  per-ticker memory log, checkpoints off. `.env` lookup is fenced (a nested worktree
  would otherwise load the main checkout's real `.env`); a scenario can supply its own.
* **Clock** — `time.sleep` is recorded, not slept (vendor pacing/backoff).

Nothing inside the fork is patched except a pass-through spy on
`TradingAgentsGraph.propagate` (the documented entry point) that records its arguments
and returned state. stdout is never written by the harness — it is the contract.

The result's `decision`, `report_path` and `plan` come from `contract_reader.py`.

## Running

Production runs Python 3.11 with `TZ=Asia/Seoul`; run the lock there:

```bash
TZ=Asia/Seoul /path/to/py311/bin/python -m pytest -q -p no:cacheprovider tests/alphapulse_contract
```

It also runs on the developer `.venv` (3.13). The subprocess uses the interpreter that
runs pytest unless `ALPHAPULSE_HARNESS_PYTHON` names another one (it must be 3.11+ as
well). Everything is offline and deterministic; the whole directory is about 40 main.py
runs and takes about a minute and a half (runs are cached per session by the `ap_run`
fixture). The sanity test proves the subprocess imports THIS checkout — pytest run from
a worktree imports the worktree even if an editable install points elsewhere.

Two environment gates, both reported as skips (with the reason) rather than failures:

* **Python < 3.11**: the whole package is skipped (`conftest.py`). Production runs
  CPython 3.11; on 3.10 main.py rejects the nightly `YYYYMMDD` date (3.10's
  `date.fromisoformat` does not read the basic format), so the fork's 3.10 CI lane
  cannot hold this contract by construction.
* **No Vertex SDK**: `test_real_vertex_sdk_sends_the_production_request_for_each_role`
  replays the recorded constructor kwargs through the REAL `ChatAnthropicVertex`; the
  fork's CI installs only `.[dev]`, so there it is skipped. Install `.[vertex]` (as
  production does) to run it. Every other test uses the harness's fake SDK class.

### Accepted drift (flipped in the upstream v0.5.1 merge)

Five tests pin behaviour the upstream v0.5.1 merge changed on purpose. They were written
against a2981a7 as `xfail(strict=True, raises=ExpectedDrift)`; the merge brought the new
behaviour in and removed the markers, so they are plain contracts now. Each asserts the
new behaviour and raises `_compat.ExpectedDrift` if a2981a7's comes back (a regression)
— `grep -rn ExpectedDrift` lists them:

| Test | a2981a7 | since the v0.5.1 merge |
|---|---|---|
| `test_alphapulse_merge_expectations::test_future_trade_date_fails_the_run_before_any_model_call_or_write` | analyses a future date | ValueError before any model call |
| `…::test_verified_snapshot_uses_the_last_settled_bar_when_the_newest_close_is_missing` | NaN newest close ⇒ rc 1 | last settled bar |
| `…::test_reflection_error_leaves_the_entry_pending_and_the_run_completes` | reflector 429 ⇒ rc 1 | entry stays pending, run completes |
| `…::test_rerun_that_settles_its_own_date_does_not_record_that_date_again` | date recorded twice | recorded once |
| `test_alphapulse_memory_contract::test_rerun_on_an_already_settled_date_does_not_record_the_decision_twice` | date recorded twice | recorded once |

For the next upstream merge, pin behaviour it is expected to change the same way:
a strict xfail on the pre-merge tree, removed in the merge commit once it passes.

## Writing a contract test

* Every test's docstring states the production change (the break) it catches.
* Expected values are hand-derived literals or reviewed fixtures — never recomputed by
  the code under test.
* Locate internals that a merge may move with `_compat.import_first(...)` /
  `get_symbol(name, ...)`, and a class a merge may also rename, which nothing of alpha-pulse
  names, by what it does with `get_class_by_interface(methods, ...)` (the memory log) —
  they FAIL (never skip) when nothing resolves. Derive preconditions from the tree (e.g.
  "every source file was walked"), never from a list of module names a merge may rename.
* Compare a JSON payload alpha-pulse reads by key (the TRADE_PLAN_JSON plan) with
  `canonical_json(project_to_expected(actual, expected)) == canonical_json(expected)`: a new
  optional key is not a break, a dropped, renamed or re-typed one (11120 for 11120.0) is.
* A behaviour that depends on the wall clock must be made deterministic by moving the
  process clock (TZ, as the missing-date test does), not left to the hour the suite runs.
* Derive paths from the test file (`Path(__file__).resolve().parents[2]` is the repo
  root); never hardcode a checkout path.
* Prefer `ap_run("<scenario>")` (cached, read-only) and `run_main(tmp_path, ...)` for a
  private or mutated scenario (`scenarios.derive(base, patch)`).
* Assert through the capture API (`RunResult.prompt_for`, `llm_for`, `tool_results`,
  `data_calls`, `config_seen`, `final_state`) and read stdout through `contract_reader`,
  not by patching fork functions.
* Keep this directory free of consumer material: no consumer code or copies of it, no
  recorded production data (plans, their numbers, dates or storyline included), no
  consumer-internal file or module names beyond what the fork's own public files already
  mention. New plan, account, news or memory fixtures are hand-written and fictional.

## When main.py or the consumer changes on purpose

1. main.py or the fork's output changed deliberately (a new stdout line, a new plan
   field): change the expected literals in the affected tests **in the same PR**, update
   docs/INTEGRATION.md §1b, and say so in the commit message. Change `contract_reader.py`
   only when the documented grammar changes, and pin the new grammar in the sanity test
   with hand-written stdout first.
2. alpha-pulse changed how it reads the output: that is re-checked in the private
   companion suite; a public test changes only if the fork-side contract moves with it.
3. Never loosen a test to make a merge green.

## The rule these tests live by

A contract test that passes on the merge because it no longer looks is worse than none.
Every test here must stay GREEN on the fork as production runs it and must go RED on the
breakage its docstring names. Check that before relying on a test: copy the tree to a
scratch directory (`rsync -a --exclude=.git --exclude=__pycache__ <repo>/ <scratch>/`),
apply the breakage there (for example upstream's `_coerce_optional_float`, the v0.5.1
last-label rating parser, a lost `position_context`, dropped Vertex kwargs), and run the
test file from the copy — it must fail. Never mutate the real checkout.

## Files

| File | Role |
|---|---|
| `harness.py` | test side: `run_main`, `RunResult`, env construction, stdout/report reading |
| `contract_reader.py` | clean-room reader of main.py's documented stdout grammar (decision, report path, plan, REVIEW) |
| `scenarios.py` | declarative scenarios s1–s5 (schema in its docstring), fixture loaders, `derive` |
| `conftest.py` | `ap_run` session cache; the Python 3.11 gate |
| `_bootstrap.py` | subprocess entry: installs fakes, spies `propagate`, runs main.py |
| `_fake_llm.py`, `_fake_data.py`, `_runtime.py` | subprocess-side fakes and capture state |
| `_compat.py` | `import_first` / `get_symbol` / `get_class_by_interface`; `project_to_expected` / `canonical_json`; `ExpectedDrift` |
| `test_alphapulse_harness_sanity.py` | the harness itself: this checkout, offline, no writes, the reader's grammar, fixture shape |
| `test_alphapulse_output_contract.py` (+ `_output_helpers.py`) | rc, stdout grammar (REVIEW for an unreadable decision), TRADE_PLAN_JSON line, report tree, dates (the missing-date default under a zone off the UTC date), main.py imports |
| `test_alphapulse_trade_plan_contract.py` | plan schema: fields, types, vocabularies, the pre-2026-08 shape rejected, all-or-nothing numbers |
| `test_alphapulse_account_contract.py` | account context reaches the PM only; scrub of the archived copy |
| `test_alphapulse_memory_contract.py` (+ `_memory_helpers.py`) | per-ticker memory log, settlement, benchmarks, checkpoints off |
| `test_alphapulse_llm_contract.py` (+ `_llm_helpers.py`) | per-role Vertex kwargs, the real SDK's request, Korean output, KR data config |
| `test_alphapulse_data_contract.py` | vendor routing, KR vendors, AV cross-check, Yahoo throttle fails the run |
| `test_alphapulse_deploy_contract.py` (+ `_deploy_helpers.py`) | pyproject/venv rebuild, 3.11 syntax, import side effects, declared dependencies |
| `test_alphapulse_merge_expectations.py` (+ `_merge_expectations_helpers.py`) | the expected-drift xfails above |
| `fixtures/` (synthetic plans) | `synthetic_sell_plan.json` (s1's Sell plan: every tranche trigger and trigger kind, full exit, kill switch), `synthetic_legacy_underweight_plan.json` (the pre-2026-08 plan shape) |
| `fixtures/` (synthetic account) | `position_contexts.json` (made-up account and the one-line JSONs built from it) |
| `fixtures/` (captured at a2981a7, reviewed) | `position_block_*.txt` (PM account block), `memory_log_a2981a7_*.md` + `memory_log_a2981a7_expected.json` (server-shaped logs and their reviewed reading) |
| `fixtures/` (harness data, fictional) | `instruments.json`, `http_routes.json`, `memory_seed_417310.KS.md` |
