"""alpha-pulse LLM contract: what each role sends to Vertex, and the config its data tools read.

alpha-pulse runs ``main.py`` unmodified, so the model side of production is exactly
what ``main.build_config()`` makes the fork build, and production pins it: the two
judges (Research Manager, Portfolio Manager) on ``claude-opus-5-5`` at effort ``xhigh``;
every other role, and the settlement reflection, on ``claude-sonnet-5-5`` at effort
``high``; ``max_tokens`` 32000 and adaptive thinking everywhere; the Vertex project from
GOOGLE_CLOUD_PROJECT (the harness sets the fork's documented example, ``tpmn-dev``) at
location ``global`` (GOOGLE_CLOUD_LOCATION left unset); Korean output; the KR vendors
(news ``naver,yfinance``, fundamentals ``wisereport,yfinance``, Naver 종목토론방) and the
KR macro-news region for .KS/.KQ.

How these tests look at it:

* Model side: main.py runs in the harness with ``llm_boundary: "sdk"`` — the fake
  stands in for ``ChatAnthropicVertex`` itself, so main.py's config, the graph's role
  resolver, the client factory and ``VertexAnthropicClient``'s kwargs translation are
  all the real code, and each record is the exact constructor call production makes.
* Wire: those constructor calls are replayed through the REAL ``ChatAnthropicVertex``
  with only the HTTP transport faked (``_llm_helpers``). Wherever the fork puts a
  setting, what counts is whether the real SDK puts it in the Vertex request.
* Data side: ``get_config()`` as the data tools saw it during the run, plus the
  library-level calls those tools made (yfinance Search queries, HTTP URLs).

Expected values are literals derived by hand from main.py at a2981a7 and from the
Vertex REST endpoint; nothing is recomputed by the code under test.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest

from . import contract_reader, scenarios
from ._llm_helpers import replay_on_real_vertex_sdk
from .harness import REPO_ROOT, RunResult, harness_python

S1 = "s1_nightly_kr_holding_sell"
S4 = "s4_us_not_held_discovery"
S5A = "s5a_nightly_yyyymmdd_not_held"
S5B = "s5b_no_date_kst_today"

# Role keys as cli.report_meta / the harness name them.
JUDGE_ROLES = ("research_manager", "portfolio_manager")
WORKER_ROLES = (
    "market_analyst", "sentiment_analyst", "news_analyst", "fundamentals_analyst",
    "bull_researcher", "bear_researcher", "trader",
    "aggressive_debator", "conservative_debator", "neutral_debator",
)
GRAPH_ROLES = JUDGE_ROLES + WORKER_ROLES
# The settlement reflection main.py triggers at run start (s1 seeds a pending entry).
REFLECTOR = "reflector"
LLM_ROLES = GRAPH_ROLES + (REFLECTOR,)


def _vertex_kwargs(model: str, effort: str) -> dict[str, Any]:
    """ChatAnthropicVertex(**kwargs) as production builds it (hand-derived from main.py).

    model_name: deep_think_llm / quick_think_llm or the role_models override;
    project: GOOGLE_CLOUD_PROJECT as production sets it; location: the 'global'
    default (GOOGLE_CLOUD_LOCATION unset in production); max_tokens:
    anthropic_max_tokens; effort and thinking ride in model_kwargs because
    ChatAnthropicVertex has no such fields and spreads model_kwargs into the request.
    """
    return {
        "model_name": model,
        "project": "tpmn-dev",
        "location": "global",
        "max_tokens": 32000,
        "model_kwargs": {"output_config": {"effort": effort}, "thinking": {"type": "adaptive"}},
    }


JUDGE_SDK_KWARGS = _vertex_kwargs("claude-opus-5-5", "xhigh")
WORKER_SDK_KWARGS = _vertex_kwargs("claude-sonnet-5-5", "high")

# Vertex AI's Anthropic endpoint: model, project and location travel in the URL.
RAW_PREDICT_URL = ("https://aiplatform.googleapis.com/v1/projects/tpmn-dev/locations/global/"
                   "publishers/anthropic/models/{model}:rawPredict")

KOSDAQ_TICKER = "247540.KQ"
KR_CODES = {"005930.KS": "005930", KOSDAQ_TICKER: "247540"}
REGION_TICKERS = ("005930.KS", KOSDAQ_TICKER, "AAPL")

# Where a data access made on behalf of an analyst is attributed: its ToolNode in the
# fork today, or the analyst's own top-level node once analysts become subgraphs.
NEWS_TOOL_NODES = {"tools_news", "News Analyst"}
FUNDAMENTALS_TOOL_NODES = {"tools_fundamentals", "Fundamentals Analyst"}
SENTIMENT_NODES = {"Sentiment Analyst", "Social Analyst", "tools_social"}

_KR_MACRO_QUERY = re.compile(r"\b(?:Korea|KOSPI|KOSDAQ)\b")
_KOREAN_OUTPUT = re.compile(r"\bKorean\b|한국어")


# ============================================================================ scenarios

def _s1_sdk() -> dict[str, Any]:
    """s1 (nightly KR holding, pending memory entry -> reflection) at the SDK boundary.

    Same derivation as the harness sanity test, so the session cache runs it once.
    """
    return scenarios.derive(scenarios.get(S1), {"llm_boundary": "sdk"}, name=S1 + "_sdk")


def _kosdaq_discovery() -> dict[str, Any]:
    """A KOSDAQ name (247540.KQ, 에코프로비엠) the instrument fixtures do not list.

    Discovery shape (explicit '' position context) on the nightly date form; the
    yfinance fake serves it from ``info_overrides`` and a synthetic price path, and
    Naver/wisereport answer through their generic "other KR" routes.
    """
    return scenarios.derive(
        scenarios.get(S5B),
        {"yfinance": {"info_overrides": {KOSDAQ_TICKER: {
            "symbol": KOSDAQ_TICKER, "shortName": "에코프로비엠", "longName": "에코프로비엠",
            "quoteType": "EQUITY", "exchange": "KOE", "currency": "KRW",
            "financialCurrency": "KRW", "country": "South Korea",
            "exchangeTimezoneName": "Asia/Seoul", "regularMarketPrice": 100.0,
        }}}},
        name="llm_kosdaq_247540_discovery",
        description="Discovery run of a KOSDAQ ticker absent from the instrument fixtures.",
        argv=[KOSDAQ_TICKER, "20260819"],
        company="에코프로비엠",
    )


def _region_run(ap_run, ticker: str) -> RunResult:
    spec = {"005930.KS": S5A, KOSDAQ_TICKER: _kosdaq_discovery(), "AAPL": S4}[ticker]
    return ap_run(spec).assert_ok()


# ============================================================================ helpers

def _under(path: str | None, root: Path) -> bool:
    return path is not None and Path(path).resolve().is_relative_to(root.resolve())


def _sdk_kwargs(record: dict[str, Any]) -> dict[str, Any]:
    """Constructor kwargs of one SDK record, minus run-global observers (callbacks)."""
    return {k: v for k, v in (record.get("kwargs") or {}).items() if k != "callbacks"}


def _key(kwargs: dict[str, Any]) -> str:
    return json.dumps(kwargs, sort_keys=True, ensure_ascii=False)


def _expected_kwargs(role: str) -> dict[str, Any]:
    return JUDGE_SDK_KWARGS if role in JUDGE_ROLES else WORKER_SDK_KWARGS


def _expected_model_effort(role: str) -> tuple[str, str]:
    return ("claude-opus-5-5", "xhigh") if role in JUDGE_ROLES else ("claude-sonnet-5-5", "high")


def _config_at_first_data_tool(res: RunResult) -> dict[str, Any]:
    cfg = res.config_seen()
    assert cfg, ("no get_config() snapshot was taken at a data tool call — no graph node "
                 f"reached the data layer?\n{res.describe()}")
    assert "__error__" not in cfg, cfg
    return cfg


def _http_calls(res: RunResult) -> list[tuple[str | None, str]]:
    return [(d.get("node"), str(d.get("url") or "")) for d in res.data_calls if d.get("lib") == "http"]


def _global_news_queries(res: RunResult) -> list[str]:
    return [str(d.get("query")) for d in res.data_calls
            if d.get("lib") == "yfinance" and d.get("api") == "Search"
            and d.get("node") in NEWS_TOOL_NODES]


def _vertex_sdk_installed(python: str) -> bool:
    """Is ``langchain_google_vertexai`` (the [vertex] extra's SDK) installed for ``python``?

    Asked the way the replay itself runs (``-I``). The fork's CI and a plain dev install
    use only ``.[dev]`` (pyproject: the Vertex SDKs are lazy-imported and optional), so
    there is no real SDK to replay through; production installs ``.[vertex]``.
    """
    probe = ("import importlib.util, sys; "
             "sys.exit(0 if importlib.util.find_spec('langchain_google_vertexai') else 3)")
    proc = subprocess.run([python, "-I", "-c", probe], capture_output=True, timeout=120)
    assert proc.returncode in (0, 3), (
        f"could not probe {python} for the Vertex SDK: rc {proc.returncode}\n"
        + proc.stderr.decode(errors="replace")[-2000:])
    return proc.returncode == 0


@pytest.fixture(scope="module")
def vertex_wire(tmp_path_factory):
    """replay(kwargs_by_key) -> {key: result}; each distinct kwargs set is replayed once.

    A result is {"key", "error", "blocked", "requests": [{"method", "url", "body"}]} from
    the real ChatAnthropicVertex (see _llm_helpers). The replay runs in the harness
    interpreter, so it sees the SDK versions the harness subprocess sees.

    Skipped (not failed) only when that interpreter has no langchain-google-vertexai at
    all -- the fork's CI installs ``.[dev]`` without the ``[vertex]`` extra. The SDK
    being present but broken still fails below; that production declares the extra is
    locked by test_alphapulse_deploy_contract.
    """
    if not _vertex_sdk_installed(harness_python()):
        pytest.skip(f"the [vertex] extra (langchain-google-vertexai) is not installed for "
                    f"{harness_python()}; the wire replay needs the real SDK "
                    "(pip install -e '.[vertex]'), production installs it")
    cache: dict[str, dict[str, Any]] = {}

    def replay(kwargs_by_key: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        missing = {k: v for k, v in kwargs_by_key.items() if k not in cache}
        if missing:
            out = replay_on_real_vertex_sdk(missing, tmp_path_factory.mktemp("vertex_wire"),
                                            python=harness_python())
            assert out.get("rc") == 0 and not out.get("import_error"), (
                "the real Vertex SDK could not be loaded for the replay (production installs "
                f"the [vertex] extra): {out.get('import_error')}\n{out.get('stderr')}")
            assert out.get("patched_transports"), out
            assert out.get("blocked_before_cases") == [], out.get("blocked_before_cases")
            for result in out.get("results") or []:
                cache[result["key"]] = {**result, "versions": out.get("versions")}
        return {k: cache[k] for k in kwargs_by_key}

    return replay


# ============================================================================ tests: model side

def test_llm_contract_runs_exercise_this_checkout(ap_run):
    """Breaks if: the harness subprocess builds its LLM clients from another copy of the fork
    (a site-packages build in the Python 3.11 venv, the developer .venv's editable install of
    the main checkout) — every verdict in this module would then be about the wrong code."""
    res = ap_run(_s1_sdk()).assert_ok()
    imports = res.captures["imports"]
    for module in ("tradingagents", "tradingagents.graph.trading_graph"):
        assert _under(imports.get(module), REPO_ROOT), (module, imports.get(module), REPO_ROOT)
    assert res.llm_constructions, "no ChatAnthropicVertex was constructed"
    for record in res.llm_constructions:
        # The harness records stack frames relative to REPO_ROOT only for files under it.
        assert any(frame.startswith("tradingagents/") for frame in record["stack"]), record["stack"]


@pytest.mark.parametrize("role", LLM_ROLES)
def test_each_role_gets_its_vertex_client_with_the_production_settings(ap_run, role):
    """Breaks if: a merge changes what a role's ``ChatAnthropicVertex`` is built with:
    role_models stops reaching the graph (upstream's ``GraphSetup(quick, deep)`` wiring or a
    dropped role resolver puts RM/PM on the deep tier: opus at effort 'high');
    upstream's ``build_llm_kwargs`` (no vertex_anthropic branch) replaces the fork's
    provider-kwargs builder (max_tokens falls back to the SDK's 4096, effort and thinking
    vanish); effort/thinking stop riding in model_kwargs or the 'adaptive' shorthand stops
    being wrapped into {'type': 'adaptive'}; project/location resolution changes; a new
    default (a temperature, an llm_max_retries) leaks into the SDK call; or the settlement
    reflection moves off the Sonnet/high client. All silent: rc stays 0."""
    res = ap_run(_s1_sdk()).assert_ok()
    record = res.llm_for(role)
    assert record["boundary"] == "sdk" and record.get("sdk_class") == "ChatAnthropicVertex", record
    assert _sdk_kwargs(record) == _expected_kwargs(role), (
        f"{role} is no longer built with the production Vertex settings (stack: {record['stack']})")


def test_generic_max_tokens_override_leaves_every_vertex_client_unchanged(ap_run):
    """Breaks if: the generic output cap (config['max_tokens'], set by TRADINGAGENTS_MAX_TOKENS,
    forwarded to every provider since upstream #1204) overrides the dedicated
    anthropic_max_tokens 32000 — the fork's ``if key not in kwargs`` guard; upstream's
    ``build_llm_kwargs`` has none. A lower cap lets the judges' thinking spend it (live,
    Opus 5.5: thinking only, stop_reason max_tokens -> no tool call -> no TRADE_PLAN_JSON);
    a higher one lengthens the worst-case call. Neither shows up as a failed run."""
    res = ap_run(_s1_sdk(), env_overrides={"TRADINGAGENTS_MAX_TOKENS": "8000"}).assert_ok()
    # Not vacuous: the generic cap really reached the config main.py built (the env string
    # stays a string there; the provider-kwargs builder coerces it).
    assert res.env_seen.get("TRADINGAGENTS_MAX_TOKENS") == "8000"
    generic_cap = (res.graph_config or {}).get("max_tokens")
    assert str(generic_cap) == "8000", (
        "TRADINGAGENTS_MAX_TOKENS no longer lands in config['max_tokens'] "
        f"(got {generic_cap!r}); re-aim this test at the generic output cap the merged code uses")
    built = {role: _sdk_kwargs(res.llm_for(role)) for role in LLM_ROLES}
    changed = {role: kwargs for role, kwargs in built.items() if kwargs != _expected_kwargs(role)}
    assert changed == {}, f"TRADINGAGENTS_MAX_TOKENS=8000 changed these Vertex clients: {changed}"


@pytest.mark.parametrize("role", LLM_ROLES)
def test_real_vertex_sdk_sends_the_production_request_for_each_role(ap_run, vertex_wire, role):
    """Breaks if: a role's constructor kwargs no longer become the production Vertex request
    in the real SDK: a setting placed where ChatAnthropicVertex does not forward it (effort
    as a top-level kwarg -> Messages.create() TypeError on every call; thinking as a bare
    string), a langchain-google-vertexai/anthropic upgrade that stops spreading model_kwargs
    into the request (the constructor call is then unchanged — only the wire shows it), a
    missing project (the SDK falls back to ADC discovery), or any model/effort/max_tokens/
    thinking/location/temperature drift, as Vertex would receive it (URL + JSON body)."""
    res = ap_run(_s1_sdk()).assert_ok()
    # Replay every client the run built in one subprocess; later parametrized cases hit
    # the cache.
    vertex_wire({_key(_sdk_kwargs(record)): _sdk_kwargs(record) for record in res.llm_constructions})

    kwargs = _sdk_kwargs(res.llm_for(role))
    result = vertex_wire({_key(kwargs): kwargs})[_key(kwargs)]
    model, effort = _expected_model_effort(role)
    assert result["error"] is None, (
        f"the real SDK ({result['versions']}) could not send {role}'s request from {kwargs}: "
        f"{result['error']} (network lookups it attempted: {result['blocked'][:3]})")
    # Complete kwargs need no ADC/metadata lookup (a missing project would trigger one, and
    # production could then resolve a different project from the ADC file).
    assert result["blocked"] == [], (role, result["blocked"][:3])
    assert len(result["requests"]) == 1, result
    request = result["requests"][0]
    assert request["method"] == "POST"
    assert request["url"] == RAW_PREDICT_URL.format(model=model), (role, request["url"])
    body = request["body"] or {}
    assert body.get("max_tokens") == 32000, (role, body.get("max_tokens"))
    assert body.get("output_config") == {"effort": effort}, (role, body.get("output_config"))
    assert body.get("thinking") == {"type": "adaptive"}, (role, body.get("thinking"))
    assert not {"temperature", "top_p", "top_k"} & set(body), (role, sorted(body))


def test_every_llm_call_of_a_nightly_run_uses_one_of_the_two_production_clients(ap_run):
    """Breaks if: a merge adds an LLM step the role checks above do not name (outside the
    graph or under a new node) on a client with other settings — e.g. upstream work on the
    deep-tier client, which main.py builds but no role uses (opus at effort 'high'), or on a
    client built without the Vertex kwargs."""
    res = ap_run(_s1_sdk()).assert_ok()
    allowed = {_key(JUDGE_SDK_KWARGS), _key(WORKER_SDK_KWARGS)}
    records = res.llm_constructions
    stray = []
    for call in res.llm_calls:
        llm_id = call.get("llm_id")
        record = records[llm_id] if isinstance(llm_id, int) and 0 <= llm_id < len(records) else {}
        if call.get("role") not in LLM_ROLES or _key(_sdk_kwargs(record)) not in allowed:
            stray.append({"role": call.get("role"), "node": call.get("node"),
                          "kind": call.get("kind"), "kwargs": record.get("kwargs")})
    assert stray == [], stray[:5]
    assert {c.get("role") for c in res.llm_calls} == set(LLM_ROLES)


def test_no_request_of_a_nightly_run_forces_a_tool_call(ap_run):
    """Breaks if: a structured call goes back to a forced ``tool_choice`` (``{"type": "tool"}``,
    ``"any"``, a tool name) — what ``ChatAnthropicVertex.with_structured_output`` sends.
    Claude Opus 5.5 / Sonnet 5.5 answer it with a 400 (the harness does the same for those
    models), the fork falls back to free text, and the run still exits 0: Research Manager,
    Trader and Portfolio Manager lose their typed output and stdout loses TRADE_PLAN_JSON."""
    res = ap_run(_s1_sdk()).assert_ok()
    forced = [(c.get("role"), c.get("tool_choice")) for c in res.llm_calls
              if c.get("tool_choice") not in (None, "auto")]
    assert forced == [], forced[:5]
    answered = {c.get("role") for c in res.llm_calls
                if c.get("kind") == "structured" and "raised" not in (c.get("reply") or {})}
    assert {"research_manager", "trader", "portfolio_manager"} <= answered, answered
    plan = contract_reader.read_trade_plan(res.stdout)
    assert plan is not None and plan.get("rating") == res.decision, (plan, res.decision)


@pytest.mark.parametrize("role", GRAPH_ROLES)
def test_every_role_is_told_to_write_in_korean(ap_run, role):
    """Breaks if: a role stops being asked for Korean output (output_language dropped from
    main.build_config(), or a merged prompt losing get_language_instruction()). Every role's
    output is a section of complete_report.md, which alpha-pulse shows to Korean readers.
    Checked on the US discovery run so no Korean market content can stand in for the
    instruction."""
    res = ap_run(S4).assert_ok()
    prompt = res.prompt_for(role)
    assert _KOREAN_OUTPUT.search(prompt), f"{role}'s prompt does not ask for Korean output"


# ============================================================================ tests: data side

@pytest.mark.parametrize("ticker", REGION_TICKERS)
def test_data_tools_see_the_macro_news_region_of_the_ticker(ap_run, ticker):
    """Breaks if: the KR macro-news region is missing from the config the data tools read
    during the run — upstream's ``run_config()`` snapshot (v0.5.1) taken before propagate()
    sets ``news_region``, a region map that loses .KQ, or the region leaking into a US run.
    A KR run would silently get US (Fed/S&P) macro news."""
    res = _region_run(ap_run, ticker)
    for where, cfg in (("first data tool call", _config_at_first_data_tool(res)),
                       ("News Analyst", res.config_seen("News Analyst"))):
        assert cfg is not None, f"no config snapshot at {where}"
        if ticker in KR_CODES:
            assert cfg.get("news_region") == "KR", (ticker, where, cfg.get("news_region"))
        else:
            assert cfg.get("news_region") != "KR", (ticker, where, cfg.get("news_region"))


@pytest.mark.parametrize("ticker", REGION_TICKERS)
def test_global_macro_news_queries_follow_the_ticker_region(ap_run, ticker):
    """Breaks if: the global-news tool stops choosing macro queries by region — a KR run must
    ask about Korea (Bank of Korea / KOSPI / KOSDAQ), a US run must not. This is what the
    region is for; the config check above cannot see a query builder that ignores it."""
    res = _region_run(ap_run, ticker)
    queries = _global_news_queries(res)
    assert queries, f"the global-news tool made no search: {res.describe()}"
    korean = [q for q in queries if _KR_MACRO_QUERY.search(q)]
    if ticker in KR_CODES:
        assert korean, (ticker, queries)
    else:
        assert korean == [], (ticker, queries)


@pytest.mark.parametrize("ticker", REGION_TICKERS)
def test_data_tools_read_alpha_pulse_vendor_chains_and_the_discussion_flag(ap_run, ticker):
    """Breaks if: main.build_config()'s KR data settings do not reach the config the data tools
    read — the news chain 'naver,yfinance', the fundamentals chain 'wisereport,yfinance', or
    enable_kr_discussion_sentiment=True (main.py resolved to upstream's, which has neither,
    or a merge dropping the data_vendors override / the discussion flag from build_config)."""
    res = _region_run(ap_run, ticker)
    cfg = _config_at_first_data_tool(res)
    vendors = cfg.get("data_vendors") or {}
    assert vendors.get("news_data") == "naver,yfinance", vendors
    assert vendors.get("fundamental_data") == "wisereport,yfinance", vendors
    assert cfg.get("enable_kr_discussion_sentiment") is True, cfg.get("enable_kr_discussion_sentiment")


@pytest.mark.parametrize("ticker", sorted(KR_CODES))
def test_kr_vendor_chains_actually_serve_kr_tickers(ap_run, ticker):
    """Breaks if: the configured KR chains are present but dead — the router lost its naver /
    wisereport registrations (the merge must union VENDOR_METHODS; an unknown vendor name is
    silently dropped from the chain), or the sentiment analyst lost its 종목토론방 pre-fetch.
    Each KR run must request Naver news from the news tool path, wisereport from the
    fundamentals tool path, and the discussion board from the sentiment analyst."""
    res = _region_run(ap_run, ticker)
    code = KR_CODES[ticker]
    calls = _http_calls(res)
    assert any(node in NEWS_TOOL_NODES and f"m.stock.naver.com/api/news/stock/{code}" in url
               for node, url in calls), calls
    assert any(node in FUNDAMENTALS_TOOL_NODES and "navercomp.wisereport.co.kr/" in url
               and f"cmp_cd={code}" in url for node, url in calls), calls
    assert any(node in SENTIMENT_NODES and "m.stock.naver.com/front-api/discussion/list" in url
               and f"itemCode={code}" in url for node, url in calls), calls


def test_kr_only_vendors_stay_inert_for_a_us_ticker(ap_run):
    """Breaks if: the KR-only vendors fire for a US ticker instead of raising before any request
    (main.py relies on that: the chain must fall through to yfinance unchanged, so US runs are
    unaffected by the KR settings)."""
    res = _region_run(ap_run, "AAPL")
    kr_hosts = [(node, url) for node, url in _http_calls(res)
                if "stock.naver.com" in url or "wisereport.co.kr" in url]
    assert kr_hosts == [], kr_hosts
    # ...and the chains still served the US ticker through yfinance.
    assert any(d.get("lib") == "yfinance" and d.get("api") == "Ticker.get_news"
               and d.get("symbol") == "AAPL" and d.get("node") in NEWS_TOOL_NODES
               for d in res.data_calls), "the news tool no longer fell through to yfinance"
