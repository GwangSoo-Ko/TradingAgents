"""alpha-pulse LLM 계약: 각 역할이 Anthropic API 에 보내는 것, 그리고 데이터 도구가 읽는 설정.

alpha-pulse 는 ``main.py`` 를 고치지 않고 돌리므로, 운영의 모델 쪽은 정확히
``main.build_config()`` 가 fork 에게 만들게 하는 것이고 운영은 그것을 못 박는다: 두 심판
(Research Manager, Portfolio Manager)은 ``claude-opus-5-5`` effort ``xhigh``, 나머지
역할과 정산 reflection 은 ``claude-sonnet-5-5`` effort ``high``, 모두 ``max_tokens`` 32000
과 adaptive thinking. 직접 Anthropic API(공급자 ``anthropic``, 운영자의
ANTHROPIC_API_KEY; 하네스는 자리표시자)이고 GCP 프로젝트·ADC·Vertex SDK 는 끼지 않는다.
한국어 출력, KR 벤더(뉴스 ``naver,yfinance``, 펀더멘털 ``wisereport,yfinance``, 네이버
종목토론방), .KS/.KQ 의 KR 거시 뉴스 지역.

이 시험들이 보는 방법:

* 모델 쪽: 하네스에서 main.py 를 ``llm_boundary: "sdk"`` 로 돌린다 — 가짜가
  ``langchain_anthropic.ChatAnthropic`` 자체를 대신하므로 main.py 설정, 그래프의 역할
  resolver, 클라이언트 factory, ``AnthropicClient`` 의 kwargs 번역,
  ``NormalizedChatAnthropic``(구조화 출력 mixin 포함)이 모두 진짜 코드이고, 각 기록은
  운영이 하는 생성자 호출 그대로다.
* 전송(wire): 그 생성자 호출을 진짜 langchain-anthropic + anthropic SDK 위의 fork
  ``NormalizedChatAnthropic`` 으로 재생한다(``_llm_helpers``, HTTP 전송층만 가짜). 일반
  호출과 구조화 호출 각각. fork 가 설정을 어디에 두든, 중요한 것은 진짜 SDK 가 그것을
  API 요청에 넣는가다. 5.5 모델에 대한 강제 tool 호출에는 거기서도 API 의 400 으로 답한다.
* 데이터 쪽: 실행 중 데이터 도구가 본 ``get_config()`` 와 그 도구들이 한 라이브러리 수준
  호출(yfinance Search 질의, HTTP URL).

기대값은 main.py 와 Anthropic Messages API 에서 손으로 끌어낸 상수이고, 시험 대상 코드로
다시 계산하지 않는다. (Vertex 를 떠나기 전까지 이 모듈은 같은 설정을 ``ChatAnthropicVertex``
위에서 고정했다. ``vertex_anthropic`` 공급자 자체는 계속 지원하며
tests/test_vertex_clients.py 와 tests/test_claude_structured_output.py 가 다룬다.)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from . import _llm_helpers, contract_reader, scenarios
from ._fake_llm import FORCED_TOOL_CHOICE_ERROR, REJECTS_FORCED_TOOL_CHOICE
from ._llm_helpers import PROBE_SCHEMA, replay_on_real_anthropic_sdk
from .harness import HARNESS_ANTHROPIC_API_KEY, REPO_ROOT, RunResult, harness_python

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
# 노드가 구조화 호출을 하는 역할 (타입 있는 출력 -> TRADE_PLAN_JSON).
STRUCTURED_ROLES = ("research_manager", "trader", "portfolio_manager")


def _direct_kwargs(model: str, effort: str) -> dict[str, Any]:
    """운영이 만드는 ChatAnthropic(**kwargs) (main.py 에서 손으로 끌어냄).

    model: deep_think_llm / quick_think_llm 또는 role_models 덮어쓰기; max_tokens:
    anthropic_max_tokens; effort: ChatAnthropic 의 ``effort`` 필드(``reasoning_effort`` 의
    alias, ``output_config.effort`` 로 나감); thinking: 'adaptive' 약식을 API 의 dict 로
    감싼 것. api_key 는 없다(ChatAnthropic 이 ANTHROPIC_API_KEY 를 읽음). base_url 도,
    project/location 도 없다.
    """
    return {
        "model": model,
        "max_tokens": 32000,
        "effort": effort,
        "thinking": {"type": "adaptive"},
    }


JUDGE_SDK_KWARGS = _direct_kwargs("claude-opus-5-5", "xhigh")
WORKER_SDK_KWARGS = _direct_kwargs("claude-sonnet-5-5", "high")

# Anthropic Messages API: 모델은 본문에, 키는 x-api-key 헤더에 실린다.
MESSAGES_URL = "https://api.anthropic.com/v1/messages"

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


@pytest.fixture(scope="module")
def anthropic_wire(tmp_path_factory):
    """replay(kwargs_by_key) -> {key: result}; 서로 다른 kwargs 묶음마다 한 번만 재생한다.

    result 는 진짜 SDK 위 fork 의 NormalizedChatAnthropic 에서 나온 {"key", "error",
    "structured_error", "structured_parsed", "blocked", "requests": [{"phase", "method",
    "url", "headers", "body", "status"}]} 이다(_llm_helpers 참고). 재생은 하네스
    인터프리터에서 돌아 하네스 하위 프로세스와 같은 SDK 판을 본다.

    건너뛰지 않는다: langchain-anthropic(과 그것이 가져오는 anthropic SDK)은 기본
    의존성이라 운영·CI 의 모든 설치에 있다. SDK 가 없거나 깨지면 여기서 실패한다(기본
    의존성으로 남는지는 test_alphapulse_deploy_contract 가 고정).
    """
    cache: dict[str, dict[str, Any]] = {}

    def replay(kwargs_by_key: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        missing = {k: v for k, v in kwargs_by_key.items() if k not in cache}
        if missing:
            out = replay_on_real_anthropic_sdk(
                missing, tmp_path_factory.mktemp("anthropic_wire"), python=harness_python(),
                repo_root=REPO_ROOT, api_key=HARNESS_ANTHROPIC_API_KEY,
                rejects_forced=sorted(REJECTS_FORCED_TOOL_CHOICE),
                forced_error=FORCED_TOOL_CHOICE_ERROR)
            assert out.get("rc") == 0 and not out.get("import_error"), (
                "the real langchain-anthropic SDK (a base dependency) could not be loaded "
                f"for the replay: {out.get('import_error')}\n{out.get('stderr')}")
            assert out.get("patched_transports"), out
            assert out.get("blocked_before_cases") == [], out.get("blocked_before_cases")
            # 이 체크아웃의 fork 클래스가 진짜 ChatAnthropic 위에 있고, 공용 구조화 출력
            # mixin 이 그보다 앞에 있다.
            fork = out["fork_class"]
            assert _under(fork["file"], REPO_ROOT), fork
            mro = fork["mro"]
            real = "langchain_anthropic.chat_models.ChatAnthropic"
            mixin = "tradingagents.llm_clients.claude_structured.ClaudeStructuredOutput"
            assert real in mro and mixin in mro and mro.index(mixin) < mro.index(real), mro
            assert all(d["served"] == "" for d in out.get("dotenv") or []), out.get("dotenv")
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
    assert res.llm_constructions, "no ChatAnthropic was constructed"
    for record in res.llm_constructions:
        # The harness records stack frames relative to REPO_ROOT only for files under it.
        assert any(frame.startswith("tradingagents/") for frame in record["stack"]), record["stack"]


def test_a_nightly_run_uses_the_direct_api_and_needs_no_gcp(ap_run):
    """깨지는 경우: 어떤 역할이 Vertex 로 돌아간다(main.build_config() 나 role_models 항목의
    ``vertex_anthropic``, 병합된 factory 의 Vertex 기본값). 그러면 실행에 운영자가 더는 갖지
    않은 GCP 프로젝트와 ADC 가 필요하다 — 운영 호스트에서 모든 분석이 첫 LLM 호출에서
    실패하거나, (환경에 옛 프로젝트가 남아 있으면) 그 프로젝트로 청구된다. 하네스는
    GOOGLE_* 를 두지 않는다. main.py 가 만드는 LLM 은 모두 직접 ChatAnthropic 이어야 하고,
    운영자가 준 키(ANTHROPIC_API_KEY)가 실행이 보는 키다."""
    res = ap_run(_s1_sdk()).assert_ok()
    classes = {r.get("sdk_class") for r in res.llm_constructions}
    assert classes == {"ChatAnthropic"}, classes
    google = sorted(k for k in res.env_seen if k.startswith(("GOOGLE_", "GCLOUD_")))
    assert google == [], google
    assert res.env_seen.get("ANTHROPIC_API_KEY") == HARNESS_ANTHROPIC_API_KEY
    cfg = res.graph_config or {}
    providers = {cfg.get("llm_provider")} | {
        spec.get("provider") for spec in (cfg.get("role_models") or {}).values()}
    assert providers == {"anthropic"}, providers


@pytest.mark.parametrize("role", LLM_ROLES)
def test_each_role_gets_its_direct_client_with_the_production_settings(ap_run, role):
    """깨지는 경우: 병합이 역할의 ``ChatAnthropic`` 생성 인자를 바꾼다: role_models 가 그래프에
    닿지 않음(upstream 의 ``GraphSetup(quick, deep)`` 배선이나 역할 resolver 가 빠지면 RM/PM
    이 깊은 계층 = opus effort 'high'); upstream 의 ``build_llm_kwargs``(anthropic 에는
    effort 만)가 fork 의 kwargs 빌더를 대신함(max_tokens 가 ChatAnthropic 모델 기본값으로,
    thinking 사라짐); 'adaptive' 약식이 {'type': 'adaptive'} 로 감싸지지 않음; 새 기본값
    (temperature, llm_max_retries, base_url)이 SDK 호출에 샘; 정산 reflection 이 Sonnet/high
    클라이언트를 떠남. 모두 조용하다: rc 는 0 그대로."""
    res = ap_run(_s1_sdk()).assert_ok()
    record = res.llm_for(role)
    assert record["boundary"] == "sdk" and record.get("sdk_class") == "ChatAnthropic", record
    assert _sdk_kwargs(record) == _expected_kwargs(role), (
        f"{role} is no longer built with the production settings (stack: {record['stack']})")


def test_generic_max_tokens_override_leaves_every_client_unchanged(ap_run):
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
    assert changed == {}, f"TRADINGAGENTS_MAX_TOKENS=8000 changed these clients: {changed}"


# 실행 환경에 남은 엔드포인트(OpenAI 호환 릴레이 등). example.invalid 는 결코 풀리지 않는다.
STRAY_BACKEND_URL = "https://example.invalid/v1"


def _stray_backend_url_run(source: str) -> tuple[dict[str, Any], dict[str, str] | None]:
    """(시나리오, env_overrides): 엔드포인트가 환경 변수로, 또는 cwd 의 .env 로 들어온다."""
    if source == "env":
        return _s1_sdk(), {"TRADINGAGENTS_LLM_BACKEND_URL": STRAY_BACKEND_URL}
    spec = scenarios.derive(_s1_sdk(), {}, name=S1 + "_sdk_dotenv_backend_url",
                            dotenv=f"TRADINGAGENTS_LLM_BACKEND_URL={STRAY_BACKEND_URL}\n")
    return spec, None


@pytest.mark.parametrize("source", ("env", "dotenv"))
def test_a_stray_backend_url_never_redirects_the_claude_calls(ap_run, anthropic_wire, source):
    """깨지는 경우: main.build_config() 가 ``backend_url`` 을 고정하지 않는다. 그 값은
    DEFAULT_CONFIG 에서 오고, DEFAULT_CONFIG 는 ``TRADINGAGENTS_LLM_BACKEND_URL`` 을 환경
    변수로도, tradingagents 가 cwd 와 그 위에서 찾는 .env/.env.enterprise 로도 받는다.
    Vertex 시절에는 resolver 가 ``vertex_*`` 에 base_url 을 주지 않아 그런 값이 남아 있어도
    아무 일이 없었다. 직접 ``anthropic`` 공급자는 그 값을 ``ChatAnthropic(base_url=...)`` 로
    쓴다: 실행의 모든 Claude 요청이 운영자의 ANTHROPIC_API_KEY 를 x-api-key 에 실은 채 그
    엔드포인트로 가서(제3자 릴레이에 키가 새거나) 모든 분석이 실패한다. 하네스의 다른
    실행은 TRADINGAGENTS_* 를 처음부터 두지 않으므로 이 경우를 보지 못한다."""
    spec, env_overrides = _stray_backend_url_run(source)
    res = ap_run(spec, env_overrides=env_overrides).assert_ok()
    # 헛돌지 않음: 그 값이 main.py 가 DEFAULT_CONFIG 를 읽는 프로세스 환경에 실제로 있었다.
    assert res.env_seen.get("TRADINGAGENTS_LLM_BACKEND_URL") == STRAY_BACKEND_URL, (
        source, res.captures.get("dotenv"))
    assert (res.graph_config or {}).get("backend_url") is None, (res.graph_config or {}).get(
        "backend_url")
    assert res.llm_constructions, "no ChatAnthropic was constructed"
    leaked = [(r.get("id"), _sdk_kwargs(r)) for r in res.llm_constructions
              if {"base_url", "anthropic_api_url"} & set(_sdk_kwargs(r))]
    assert leaked == [], leaked
    built = {role: _sdk_kwargs(res.llm_for(role)) for role in LLM_ROLES}
    changed = {role: kwargs for role, kwargs in built.items() if kwargs != _expected_kwargs(role)}
    assert changed == {}, changed
    # 진짜 SDK 에서도: 이 실행이 만든 모든 클라이언트의 모든 요청(일반, 구조화)이 API 로 간다.
    replayed = anthropic_wire({_key(_sdk_kwargs(r)): _sdk_kwargs(r) for r in res.llm_constructions})
    urls = {request["url"] for result in replayed.values() for request in result["requests"]}
    assert urls == {MESSAGES_URL}, urls
    assert all(result["error"] is None for result in replayed.values()), replayed


@pytest.mark.parametrize("role", LLM_ROLES)
def test_real_sdk_sends_the_production_request_for_each_role(ap_run, anthropic_wire, role):
    """깨지는 경우: 역할의 생성자 kwargs 가 진짜 SDK 에서 운영 API 요청이 되지 않는다:
    ChatAnthropic 이 넘기지 않는 자리에 둔 설정(effort 를 model_kwargs 에 -> API 가 거절하는
    최상위 'effort'; thinking 을 문자열로), ``effort`` 를 ``output_config.effort`` 로 옮기지
    않거나 thinking 꼴을 바꾸는 langchain-anthropic/anthropic 업그레이드(생성자 호출은 그대로라
    wire 에서만 보인다), 32000 토큰 요청을 보내기 전에 거절하는 SDK 비스트리밍 가드
    ("Streaming is required": 실행의 모든 호출 실패), x-api-key 에 닿지 않는 키, 그리고 API 가
    받는 그대로(URL, 헤더, JSON 본문)의 model/effort/max_tokens/thinking/temperature 차이."""
    res = ap_run(_s1_sdk()).assert_ok()
    # 실행이 만든 모든 클라이언트를 하위 프로세스 하나에서 재생한다; 뒤의 매개변수 경우는
    # 캐시를 쓴다.
    anthropic_wire({_key(_sdk_kwargs(r)): _sdk_kwargs(r) for r in res.llm_constructions})

    kwargs = _sdk_kwargs(res.llm_for(role))
    result = anthropic_wire({_key(kwargs): kwargs})[_key(kwargs)]
    model, effort = _expected_model_effort(role)
    assert result["error"] is None, (
        f"the real SDK ({result['versions']}) could not send {role}'s request from {kwargs}: "
        f"{result['error']} (network lookups it attempted: {result['blocked'][:3]})")
    assert result["blocked"] == [], (role, result["blocked"][:3])
    chat = [r for r in result["requests"] if r["phase"] == "chat"]
    assert len(chat) == 1, result
    request = chat[0]
    assert request["method"] == "POST"
    assert request["url"] == MESSAGES_URL, (role, request["url"])
    assert request["headers"].get("x-api-key") == HARNESS_ANTHROPIC_API_KEY, role
    assert request["headers"].get("anthropic-version"), request["headers"]
    assert "authorization" not in request["headers"], role  # Google OAuth bearer 없음
    body = request["body"] or {}
    assert body.get("model") == model, (role, body.get("model"))
    assert body.get("max_tokens") == 32000, (role, body.get("max_tokens"))
    assert body.get("output_config") == {"effort": effort}, (role, body.get("output_config"))
    assert body.get("thinking") == {"type": "adaptive"}, (role, body.get("thinking"))
    assert not {"effort", "reasoning_effort"} & set(body), (role, sorted(body))
    assert not {"temperature", "top_p", "top_k"} & set(body), (role, sorted(body))
    assert body.get("stream") in (None, False), role  # 비스트리밍 요청으로 보냄


@pytest.mark.parametrize("role", LLM_ROLES)
def test_real_sdk_structured_call_lets_the_model_choose_the_tool(ap_run, anthropic_wire, role):
    """깨지는 경우: 운영이 만드는 클래스가 진짜 SDK 에서 구조화 호출을 ``tool_choice`` auto 로
    보내지 않는다 — ``NormalizedChatAnthropic`` 에서 공용 ``ClaudeStructuredOutput`` mixin 이
    빠짐(그러면 ChatAnthropic 자신의 메서드가 thinking 이 켜져 있으면 tool_choice 없이, 다시
    묻지도 않고 보내고, thinking 이 없으면 tool 을 강제한다), 또는 어디서든 강제 선택.
    Claude Opus 5.5 / Sonnet 5.5 에서 강제 호출은 API 의 400 이다(재생 전송층도 같게 답함).
    fork 는 rc 0 으로 자유 텍스트에 떨어지고 stdout 에서 TRADE_PLAN_JSON 이 사라진다.
    구조화 요청도 역할의 운영 설정을 그대로 싣는다."""
    res = ap_run(_s1_sdk()).assert_ok()
    anthropic_wire({_key(_sdk_kwargs(r)): _sdk_kwargs(r) for r in res.llm_constructions})
    kwargs = _sdk_kwargs(res.llm_for(role))
    result = anthropic_wire({_key(kwargs): kwargs})[_key(kwargs)]
    model, effort = _expected_model_effort(role)
    assert result["structured_error"] is None, (role, result["structured_error"])
    assert result["structured_parsed"] == {"ok": True}, (role, result["structured_parsed"])
    structured = [r for r in result["requests"] if r["phase"] == "structured"]
    assert len(structured) == 1, structured  # 첫 요청에서 답을 받음
    body = structured[0]["body"] or {}
    assert structured[0]["status"] == 200, structured[0]
    assert body.get("tool_choice") == {"type": "auto"}, (role, body.get("tool_choice"))
    assert [t.get("name") for t in body.get("tools") or []] == [PROBE_SCHEMA], body.get("tools")
    assert (body.get("model"), body.get("max_tokens"), body.get("output_config"),
            body.get("thinking")) == (model, 32000, {"effort": effort}, {"type": "adaptive"})


def test_the_replay_transport_rejects_a_forced_tool_call_like_the_api(tmp_path):
    """재생의 400 이 실제로 작동한다(그래서 위의 구조화 확인이 우연히 통과할 수 없다): 진짜
    SDK 자신의 강제 호출 — thinking 없는 ChatAnthropic.with_structured_output — 을 운영
    모델에 보내면 400 을 받고 오류로 드러난다."""
    import subprocess

    probe = (
        "import importlib.util, json, sys\n"
        "spec = importlib.util.spec_from_file_location('replay_helpers', sys.argv[1])\n"
        "h = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(h)\n"
        "h._guard_sockets([])\n"
        "reqs, phase = [], ['forced']\n"
        "h._patch_http_transports(reqs, phase, set(json.loads(sys.argv[2])), sys.argv[3])\n"
        "from langchain_anthropic import ChatAnthropic\n"
        "from pydantic import BaseModel\n"
        "class ContractProbe(BaseModel):\n"
        "    ok: bool\n"
        "llm = ChatAnthropic(model='claude-opus-5-5', max_tokens=1024, api_key='k')\n"
        "try:\n"
        "    llm.with_structured_output(ContractProbe).invoke('ping')\n"
        "    print(json.dumps({'error': None, 'reqs': reqs}))\n"
        "except Exception as exc:\n"
        "    print(json.dumps({'error': type(exc).__name__ + ': ' + str(exc), 'reqs': reqs}))\n"
    )
    proc = subprocess.run(
        [harness_python(), "-I", "-B", "-c", probe, str(Path(_llm_helpers.__file__).resolve()),
         json.dumps(sorted(REJECTS_FORCED_TOOL_CHOICE)), FORCED_TOOL_CHOICE_ERROR],
        cwd=str(tmp_path), capture_output=True, timeout=300,
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "LANG": "C.UTF-8"})
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")[-3000:]
    out = json.loads(proc.stdout.decode().strip().splitlines()[-1])
    assert out["reqs"] and out["reqs"][0]["body"]["tool_choice"]["type"] == "tool", out
    assert out["reqs"][0]["status"] == 400, out
    assert out["error"] and FORCED_TOOL_CHOICE_ERROR in out["error"], out


def test_every_llm_call_of_a_nightly_run_uses_one_of_the_two_production_clients(ap_run):
    """Breaks if: a merge adds an LLM step the role checks above do not name (outside the
    graph or under a new node) on a client with other settings — e.g. upstream work on the
    deep-tier client, which main.py builds but no role uses (opus at effort 'high'), or on a
    client built without the production kwargs."""
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
    """깨지는 경우: 구조화 호출이 강제 ``tool_choice``(``{"type": "tool"}``, ``"any"``, tool
    이름)로 돌아간다 — thinking 없는 ``ChatAnthropic.with_structured_output`` 이 보내는 것 —
    또는 fork 의 auto-tool 경로를 떠난다(thinking 이 켜진 ChatAnthropic 자신의 메서드는
    tool_choice 없이 보내고 다시 묻지 않는다: 공용 mixin 을 잃은 것). Claude Opus 5.5 /
    Sonnet 5.5 는 강제 호출에 400 으로 답하고(하네스도 그 모델에 같게 답함), fork 는 자유
    텍스트로 떨어지며 실행은 그래도 0 으로 끝난다: Research Manager, Trader, Portfolio
    Manager 가 타입 있는 출력을 잃고 stdout 에서 TRADE_PLAN_JSON 이 사라진다."""
    res = ap_run(_s1_sdk()).assert_ok()
    forced = [(c.get("role"), c.get("tool_choice")) for c in res.llm_calls
              if c.get("tool_choice") not in (None, "auto")]
    assert forced == [], forced[:5]
    structured = [(c.get("role"), c.get("tool_choice")) for c in res.llm_calls
                  if c.get("kind") == "structured"]
    assert structured and all(choice == "auto" for _, choice in structured), structured
    answered = {c.get("role") for c in res.llm_calls
                if c.get("kind") == "structured" and "raised" not in (c.get("reply") or {})}
    assert set(STRUCTURED_ROLES) <= answered, answered
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
