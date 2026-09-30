"""Declarative scenarios for the alpha-pulse contract harness (pure data, JSON-able).

A scenario is a plain dict; ``harness.run_main(tmp_path, scenario)`` runs it.

    name              str   run-directory name
    description       str
    argv              list  what alpha-pulse passes after main.py: [TICKER] or [TICKER, DATE]
    position_context  str | None   value of TRADINGAGENTS_POSITION_CONTEXT;
                                   None = variable absent, "" = explicitly empty
    env               {name: str | None}   extra env (None removes a base variable)
    memory_log_seed   str | None   written to the per-ticker memory log before the run
    dotenv            str | None   served as the .env find_dotenv(usecwd=True) finds
    llm_boundary      "factory" (fake create_llm_client) | "sdk" (fake ChatAnthropicVertex)
    roles             {role: RoleScript}   role keys as in cli.report_meta + "reflector"
    http_rules        [rule]   tried before fixtures/http_routes.json (same rule schema)
    yfinance          {"faults": [fault], "info_overrides": {symbol: {info keys}}}
    alpha_vantage     {"extra_rows": [{"date": "YYYY-MM-DD", "close": float}]}
    real_sleep        bool     default False: time.sleep is recorded, not slept
    timeout_s         int

RoleScript:
    tool_rounds  [[{"name": tool, "args": {...}}, ...], ...]   analysts only; calls to
                 tools the node did not bind are skipped (recorded as skipped_tool_calls)
    text         str | [str]   chat replies (and the free-text fallback of a structured role)
    structured   dict | "fail" | "raise"   raw tool args for the role's schema (validated by the
                 fork's real pydantic model), prose instead of the tool, or a provider error
    structured_fail_text  str  the prose returned when structured == "fail"

Strings in scripts may use ${ticker}, ${company}, ${trade_date}, ${trade_date-N}
(N days before the trade date that reached propagate).

fault (yfinance): {"api": "history|download|info|Search|get_news|statements|insider_transactions|*",
                   "symbol": fnmatch pattern (for Search: the query), "name": str (echoed in
                   captures), "times": int (apply to the first N matching calls only), and one of
                   "raise": "YFRateLimitError|ConnectionError|HTTPError|Timeout|TimeoutError|..."
                   | "empty": true | "nan_last_close": true (newest returned bar has no Close)
                   | "drop_dates": ["YYYY-MM-DD", ...] (no bar on those days)
                   | "drop_last_bars": N (newest N bars of the requested window; for the 5-year
                   download that is today, i.e. a lag on live runs)}
                   Note: real yfinance (1.x, production's) ``yf.download`` swallows per-ticker
                   errors, rate limits included, into an empty frame -- model a failing download
                   with "empty": true; "raise" on "download" is not something the library does.
http rule: {"name", "match": [substrings of the full URL incl. query], "status": int,
            "json": obj | "text": str, "headers": {...}, "times": int,
            "raise": "ConnectionError|Timeout|ReadTimeout|URLError|TimeoutError|OSError", "message"}
alpha_vantage.extra_rows: rows added to the TIME_SERIES_DAILY CSV (e.g. a close newer than the
            lagging yfinance feed); only used when a test sets ALPHA_VANTAGE_API_KEY.
self_test_network: bool   the bootstrap first attempts real socket/curl_cffi I/O (all refused).

Use ``get(name)`` for a private deep copy and ``derive(base, patch)`` to change it.
"""

from __future__ import annotations

import copy
import json
from functools import cache
from pathlib import Path
from typing import Any

FIXTURES = Path(__file__).resolve().parent / "fixtures"
DELETE = object()  # derive(): remove this key

# The five TradingAgents ratings alpha-pulse recognises + the REVIEW sentinel.
RATINGS = ("Buy", "Overweight", "Hold", "Underweight", "Sell")


# ============================================================================ fixtures

def fixture_text(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


@cache
def _fixture_json(name: str) -> Any:
    return json.loads(fixture_text(name))


def fixture_json(name: str) -> Any:
    return copy.deepcopy(_fixture_json(name))


def sell_plan_json() -> str:
    """s1's Sell plan as the TRADE_PLAN_JSON payload (synthetic, hand-written fixture: every
    tranche trigger, every TrancheTrigger kind, a full exit and a kill switch; fictional
    numbers). It has no ``revision`` key -- the line shape of before PlanRevision."""
    return fixture_text("synthetic_sell_plan.json").rstrip("\n")


def sell_plan() -> dict[str, Any]:
    return json.loads(sell_plan_json())


def position_context(name: str) -> str:
    """'holding_founding_buy' | 'not_held' | 'explicit_empty' — the exact env value."""
    return fixture_json("position_contexts.json")[name]["env_value"]


# ============================================================================ helpers

def derive(base: dict[str, Any], patch: dict[str, Any], **top_level: Any) -> dict[str, Any]:
    """Deep-merge ``patch`` (and keyword top-level keys) into a copy of ``base``.

    Dicts merge recursively; any other value (lists included) replaces; the
    ``DELETE`` sentinel removes a key.
    """
    out = copy.deepcopy(base)

    def merge(dst: dict[str, Any], src: dict[str, Any]) -> None:
        for key, value in src.items():
            if value is DELETE:
                dst.pop(key, None)
            elif isinstance(value, dict) and isinstance(dst.get(key), dict):
                merge(dst[key], value)
            else:
                dst[key] = copy.deepcopy(value)

    merge(out, patch)
    merge(out, top_level)
    return out


def _tool(name: str, **args: Any) -> dict[str, Any]:
    return {"name": name, "args": args}


# ============================================================================ role scripts

MARKET_TOOL_ROUNDS = [
    [_tool("get_stock_data", symbol="${ticker}", start_date="${trade_date-90}", end_date="${trade_date}"),
     _tool("get_indicators", symbol="${ticker}", indicator="close_50_sma", curr_date="${trade_date}",
           look_back_days=10)],
    [_tool("get_verified_market_snapshot", symbol="${ticker}", curr_date="${trade_date}", look_back_days=10)],
]
NEWS_TOOL_ROUNDS = [[
    _tool("get_news", ticker="${ticker}", start_date="${trade_date-7}", end_date="${trade_date}"),
    _tool("get_global_news", curr_date="${trade_date}", look_back_days=7, limit=5),
    _tool("get_macro_indicators", indicator="fed_funds_rate", curr_date="${trade_date}"),
    _tool("get_prediction_markets", topic="central bank rate decision"),
]]
FUNDAMENTALS_TOOL_ROUNDS = [[
    _tool("get_fundamentals", ticker="${ticker}", curr_date="${trade_date}"),
    _tool("get_balance_sheet", ticker="${ticker}", freq="quarterly", curr_date="${trade_date}"),
    _tool("get_income_statement", ticker="${ticker}", freq="quarterly", curr_date="${trade_date}"),
    _tool("get_cashflow", ticker="${ticker}", freq="quarterly", curr_date="${trade_date}"),
]]

# Hostile lines in the debug trace (main.py runs with debug=True, so analyst output
# is printed before main.py's own contract lines). alpha-pulse must still take the
# LAST "Report saved:" line and the decision line right before the plan line.
TRACE_DECOYS = "\n\nHold\nReport saved: /tmp/decoy/complete_report.md"


def _analyst_roles(story: dict[str, str]) -> dict[str, Any]:
    return {
        "market_analyst": {
            "tool_rounds": MARKET_TOOL_ROUNDS,
            "text": ("## 기술적 분석 — ${company} (${ticker})\n\n"
                     f"- {story['market']}\n- 검증 스냅샷을 가격 수치의 기준으로 삼았다.\n\n"
                     "| 지표 | 해석 |\n|---|---|\n| 50SMA | 지지선 |\n| RSI | 중립권 |"
                     + TRACE_DECOYS),
        },
        "sentiment_analyst": {
            "structured": {
                "overall_band": story.get("band", "Mildly Bullish"),
                "overall_score": 6.1,
                "confidence": "medium",
                "narrative": (f"{story['sentiment']}\n\n| 신호 | 방향 | 출처 |\n|---|---|---|\n"
                              "| 커뮤니티 반응 | 강세 우위 | 게시판 |\n| 뉴스 흐름 | 중립 | 뉴스 |"),
            },
            "text": f"## 투자심리 — ${{company}}\n\n{story['sentiment']}",
        },
        "news_analyst": {
            "tool_rounds": NEWS_TOOL_ROUNDS,
            "text": ("## 뉴스·거시 — ${company} (${ticker})\n\n"
                     f"- {story['news']}\n\n| 이벤트 | 영향 |\n|---|---|\n| 금리 결정 | 할인율 |"),
        },
        "fundamentals_analyst": {
            "tool_rounds": FUNDAMENTALS_TOOL_ROUNDS,
            "text": ("## 펀더멘털 — ${company} (${ticker})\n\n"
                     f"{story['fundamentals']}\n\n| 항목 | 해석 |\n|---|---|\n| 밸류에이션 | 적정 |"),
        },
        "bull_researcher": {"text": story["bull"]},
        "bear_researcher": {"text": story["bear"]},
        "aggressive_debator": {"text": story["aggressive"]},
        "conservative_debator": {"text": story["conservative"]},
        "neutral_debator": {"text": story["neutral"]},
        "reflector": {"text": ("판단 방향은 벤치마크 대비 소폭 우위였다. 이벤트 일정이 수익 대부분을 설명했다. "
                               "다음에는 공시 일정을 먼저 확인한다.")},
    }


# Fictional storyline of the KR holding (sector-level themes, no event of the listed company).
REIT_STORY = {
    "market": "50일선 위에서 거래되지만 RSI가 58에서 꺾여 상승 탄력이 둔화됐다.",
    "sentiment": "종목토론방은 배당 매력을 말하는 글과 금리 부담을 걱정하는 글이 엇갈린다.",
    "news": "상장 리츠의 차입금 차환 금리 상승 보도와 금리 결정을 앞둔 섹터 변동성 확대 기사.",
    "fundamentals": "wisereport 컨센서스와 재무제표 기준 PBR 1.1배 안팎, 배당수익률 6%대.",
    "bull": "배당수익률이 섹터 평균보다 높고 순자산 대비 할인이 남아 있다.",
    "bear": "차환 금리가 오르면 배당 여력이 줄어든다. 반등은 비중을 줄일 기회다.",
    "aggressive": "50%를 즉시 처분하고 잔여는 반등 구간에서 나눠 판다.",
    "conservative": "60일선 아래로 마감하면 전량 정리한다. 금리 부담이 커지는 동안 더 들고 갈 이유가 없다.",
    "neutral": "분할 처분과 킬스위치를 함께 쓰는 절충안이 합리적이다.",
}
SAMSUNG_STORY = {
    "market": "20일선과 50일선 위에서 거래되며 거래량이 늘고 있다.",
    "sentiment": "개인 투자자 게시판은 HBM 공급 확대 기대가 우세하다.",
    "news": "HBM3E 12단 공급 확대와 외국인 5거래일 연속 순매수.",
    "fundamentals": "wisereport 컨센서스 목표가 86,667원, 순현금 구조.",
    "bull": "메모리 업황 반등과 외국인 순매수 전환이 확인됐다.",
    "bear": "밸류에이션 재평가가 이미 상당 부분 진행됐다.",
    "aggressive": "첫 트랜치를 크게 가져가 업황 반등 초기를 잡자.",
    "conservative": "200일선 이탈 시 전량 정리 조건을 둬야 한다.",
    "neutral": "두 번에 나눠 진입하는 계획이 균형적이다.",
}
APPLE_STORY = {
    "market": "50일선 위에서 거래되며 230달러 부근이 지지선이다.",
    "sentiment": "StockTwits 는 강세 1건, 약세 1건으로 엇갈린다.",
    "news": "아이폰 생산 확대와 서비스 매출 사상 최대 보도.",
    "fundamentals": "서비스 부문 마진 확대, 순이익률 24%.",
    "bull": "서비스 매출 성장과 AI 기기 교체 수요가 겹친다.",
    "bear": "PER 35배는 성장 둔화 위험을 반영하지 않는다.",
    "aggressive": "목표 비중까지 빠르게 채우자.",
    "conservative": "200일선 이탈 시 전량 정리 조건을 둬야 한다.",
    "neutral": "두 번에 나눠 진입하는 계획이 균형적이다.",
}


def _decision_roles(research: dict, trader: dict, pm: Any, pm_text: str) -> dict[str, Any]:
    return {
        "research_manager": {"structured": research,
                             "text": f"**Recommendation**: {research['recommendation']}\n\n(free text)"},
        "trader": {"structured": trader,
                   "text": f"**Action**: {trader['action']}\n\nFINAL TRANSACTION PROPOSAL: "
                           f"**{trader['action'].upper()}**"},
        "portfolio_manager": {"structured": pm, "text": pm_text},
    }


# ============================================================================ plans

REIT_RESEARCH = {
    "recommendation": "Underweight",
    "rationale": "강세 논거(배당 매력)는 금리 경로에 달려 있고, 약세 논거(차환 비용)는 이미 숫자로 확인된다.",
    "strategic_actions": "보유분의 50%를 먼저 줄이고 나머지는 조건부로 정리한다.",
}
REIT_TRADER = {
    "action": "Sell",
    "reasoning": "차환 비용이 배당 여력을 줄일 위험이 크다. 60일선 부근 10,380원이 손절선이다.",
    "entry_price": 10800.0,
    "stop_loss": 10380.0,
    "position_sizing": "보유분의 50% 즉시, 나머지 조건부",
}
# Quotes account figures on purpose (held_qty 1,250 / avg 10,480 / cash 421,675,870 in
# fixtures/position_contexts.json): the archived memory-log copy must scrub them while the
# saved report keeps the model's wording. The first tranche it names is the Sell plan's.
REIT_SELL_EXEC_SUMMARY = ("보유 1,250주(평단 10,480원)를 세 단계로 전량 처분한다. 1차 50%는 즉시 "
                          "10,720~10,860원에서 집행하고, 현금 421,675,870원은 대기 자금으로 둔다.")
REIT_SELL_THESIS = ("반등은 보유를 늘릴 이유가 아니라 비중을 줄일 기회다. 진입 논지였던 배당 여력은 "
                    "차환 금리 상승으로 약해졌다.")
REIT_REVISION = {"kind": "new_information",
                 "note": "차입금 차환 금리가 진입 당시 가정보다 높게 정해져, 기대한 배당 여력이 줄었다."}


def reit_sell_pm() -> dict[str, Any]:
    """PM raw tool args for s1: the synthetic Sell plan + prose + a revision."""
    plan = sell_plan()
    plan.update(executive_summary=REIT_SELL_EXEC_SUMMARY, investment_thesis=REIT_SELL_THESIS,
                revision=REIT_REVISION)
    return plan


PM_FREE_TEXT = ("## 최종 판단\n\n매수(Buy) 의견은 금리 경로의 불확실성 때문에 기각한다. Hold 로 버티기보다 "
                "Sell 로 정리하는 편이 낫다.")

SAMSUNG_RESEARCH = {"recommendation": "Buy",
                    "rationale": "업황 반등의 증거가 약세 논거보다 구체적이다.",
                    "strategic_actions": "2% 비중까지 두 번에 나눠 진입한다."}
SAMSUNG_TRADER = {"action": "Buy", "reasoning": "HBM 공급 확대와 수급 개선.", "entry_price": 71000.0,
                  "stop_loss": 66000.0, "position_sizing": "목표 비중 2%"}
SAMSUNG_BUY_PM = {
    "rating": "Buy",
    "executive_summary": "HBM3E 공급 확대를 근거로 2% 비중까지 두 번에 나눠 진입한다. 손절은 66,000원.",
    "investment_thesis": "메모리 업황 반등과 외국인 순매수 전환이 확인됐다. 순현금 구조가 하방을 받친다.",
    "price_target": 84000.0,
    "time_horizon": "3-6개월",
    "total_weight_pct": 2.0,
    "stop_loss": 66000.0,
    "tranches": [
        {"seq": 1, "pct": 60.0, "price_low": 70500.0, "price_high": 71500.0, "trigger": "immediate",
         "triggers": [], "condition": None},
        {"seq": 2, "pct": 40.0, "price_low": 68000.0, "price_high": 69000.0, "trigger": "conditional",
         "triggers": [{"kind": "event", "price": None, "trail_pct": None, "reference_price": 68500.0,
                       "reference_label": "20일선", "condition": "20일선 지지 확인 후 이틀 연속 종가 유지"}],
         "condition": "20일선 부근 눌림목"},
    ],
    "exit_target": None,
    "kill_switch": {"price": 64000.0, "condition": "종가 기준 200일선 이탈 시 전량 정리"},
    "revision": None,
}

APPLE_RESEARCH = {"recommendation": "Overweight",
                  "rationale": "서비스 성장이 밸류에이션 부담을 상쇄한다.",
                  "strategic_actions": "3% 비중까지 분할 진입한다."}
APPLE_TRADER = {"action": "Buy", "reasoning": "서비스 마진 확대.", "entry_price": 231.0,
                "stop_loss": 214.0, "position_sizing": "목표 비중 3%"}
APPLE_OVERWEIGHT_PM = {
    "rating": "Overweight",
    "executive_summary": "서비스 마진 확대를 근거로 목표 비중 3%까지 두 번에 나눠 진입한다. 손절은 214달러.",
    "investment_thesis": "서비스 매출이 사상 최대를 기록했고 AI 기기 교체 수요가 보인다.",
    "price_target": 255.0,
    "time_horizon": "3-6개월",
    "total_weight_pct": 3.0,
    "stop_loss": 214.0,
    "tranches": [
        {"seq": 1, "pct": 50.0, "price_low": 229.0, "price_high": 233.0, "trigger": "immediate",
         "triggers": [], "condition": None},
        {"seq": 2, "pct": 50.0, "price_low": 219.0, "price_high": 223.0, "trigger": "conditional",
         "triggers": [{"kind": "event", "price": None, "trail_pct": None, "reference_price": 221.0,
                       "reference_label": "50SMA", "condition": "50일선 지지 확인 후 이틀 연속 종가 유지"}],
         "condition": "50일선 부근 눌림목"},
    ],
    "exit_target": None,
    "kill_switch": {"price": 209.0, "condition": "종가 기준 200일선 이탈 시 전량 정리"},
    "revision": None,
}


# ============================================================================ scenarios

def _reit_base(name: str, description: str) -> dict[str, Any]:
    roles = _analyst_roles(REIT_STORY)
    roles.update(_decision_roles(REIT_RESEARCH, REIT_TRADER, reit_sell_pm(), PM_FREE_TEXT))
    return {
        "name": name,
        "description": description,
        "argv": ["417310.KS", "20260819"],
        "position_context": position_context("holding_founding_buy"),
        "env": {},
        "memory_log_seed": None,
        "dotenv": None,
        "llm_boundary": "factory",
        "roles": roles,
        "http_rules": [],
        "yfinance": {"faults": [], "info_overrides": {}},
        "real_sleep": False,
    }


def _s1() -> dict[str, Any]:
    s = _reit_base(
        "s1_nightly_kr_holding_sell",
        "Nightly batch, KR holding 417310.KS bought on a Buy thesis; argv date in the nightly "
        "KST 'YYYYMMDD' form; PM returns the synthetic Sell plan plus a revision; the "
        "per-ticker memory log holds one resolved and one pending entry (settled at run start).",
    )
    s["memory_log_seed"] = fixture_text("memory_seed_417310.KS.md")
    return s


def _s2_pm(rating: str) -> dict[str, Any]:
    pm = reit_sell_pm()
    pm["executive_summary"] = REIT_SELL_EXEC_SUMMARY + " Most recent prior rating: Buy."
    pm["investment_thesis"] = ("진입 당시 등급(Rating: Buy)의 전제였던 배당 여력이 차환 금리 상승으로 "
                               "줄었다. " + REIT_SELL_THESIS)
    pm["revision"] = {"kind": "thesis_error",
                      "note": "The founding rating - Overweight - assumed a dividend capacity that "
                              "the higher refinancing rate no longer supports."}
    pm["kill_switch"]["condition"] = pm["kill_switch"]["condition"] + " (consensus rating: Hold)"
    if rating == "Underweight":
        pm["rating"] = "Underweight"
        pm["exit_target"] = {"kind": "weight", "remaining_weight_pct": 1.0}
    return pm


def _s2() -> dict[str, Any]:
    s = _reit_base(
        "s2_pm_prose_quotes_other_ratings",
        "Same holding as s1, but the PM's prose quotes other ratings ('Most recent prior rating: "
        "Buy.', revision note 'The founding rating - Overweight - ...', kill switch "
        "'consensus rating: Hold', Korean '진입 당시 등급(Rating: Buy)'); the typed rating is Sell.",
    )
    s["roles"]["portfolio_manager"]["structured"] = _s2_pm("Sell")
    return s


def _s2u() -> dict[str, Any]:
    s = _reit_base(
        "s2u_pm_prose_quotes_other_ratings_underweight",
        "s2 with a typed Underweight (exit_target weight 1.0%).",
    )
    s["roles"]["portfolio_manager"]["structured"] = _s2_pm("Underweight")
    return s


def _s3() -> dict[str, Any]:
    s = _reit_base(
        "s3_pm_structured_failure_fallback",
        "Same holding as s1, but the PM's structured call fails (the model answers in prose), so "
        "the fork's free-text fallback runs: no TRADE_PLAN_JSON; the prose names several ratings "
        "without a 'Rating:' label.",
    )
    s["roles"]["portfolio_manager"]["structured"] = "fail"
    s["roles"]["portfolio_manager"]["structured_fail_text"] = "계획을 도구 없이 산문으로 답합니다."
    return s


def _s4() -> dict[str, Any]:
    roles = _analyst_roles(APPLE_STORY)
    roles.update(_decision_roles(APPLE_RESEARCH, APPLE_TRADER, APPLE_OVERWEIGHT_PM, PM_FREE_TEXT))
    return {
        "name": "s4_us_not_held_discovery",
        "description": "Discovery/web run of a US name (AAPL) with an ISO date and the position "
                       "context explicitly set to '' (not injected). KR-only vendors must fall "
                       "through to yfinance; PM returns a fresh-entry Overweight plan.",
        "argv": ["AAPL", "2026-08-19"],
        "position_context": position_context("explicit_empty"),
        "env": {},
        "memory_log_seed": None,
        "dotenv": None,
        "llm_boundary": "factory",
        "roles": roles,
        "http_rules": [],
        "yfinance": {"faults": [], "info_overrides": {}},
        "real_sleep": False,
    }


def _samsung_base(name: str, description: str, argv: list[str], ctx: str) -> dict[str, Any]:
    roles = _analyst_roles(SAMSUNG_STORY)
    roles.update(_decision_roles(SAMSUNG_RESEARCH, SAMSUNG_TRADER, SAMSUNG_BUY_PM, PM_FREE_TEXT))
    return {
        "name": name,
        "description": description,
        "argv": argv,
        "position_context": ctx,
        "env": {},
        "memory_log_seed": None,
        "dotenv": None,
        "llm_boundary": "factory",
        "roles": roles,
        "http_rules": [],
        "yfinance": {"faults": [], "info_overrides": {}},
        "real_sleep": False,
    }


def _s5a() -> dict[str, Any]:
    return _samsung_base(
        "s5a_nightly_yyyymmdd_not_held",
        "Nightly candidate 005930.KS (not held: account JSON with held_qty 0, absent reason "
        "not_held) with the nightly KST 'YYYYMMDD' date; propagate must receive the ISO date.",
        ["005930.KS", "20260819"], position_context("not_held"))


def _s5b() -> dict[str, Any]:
    return _samsung_base(
        "s5b_no_date_kst_today",
        "argv without a date — alpha-pulse appends the date only when it has one (its "
        "web/discovery callers pass the ISO date today, so this is the fallback path): "
        "main.py must default to today's date in TZ=Asia/Seoul; position context explicitly "
        "'' as discovery sends it.",
        ["005930.KS"], position_context("explicit_empty"))


_BUILDERS = {
    "s1_nightly_kr_holding_sell": _s1,
    "s2_pm_prose_quotes_other_ratings": _s2,
    "s2u_pm_prose_quotes_other_ratings_underweight": _s2u,
    "s3_pm_structured_failure_fallback": _s3,
    "s4_us_not_held_discovery": _s4,
    "s5a_nightly_yyyymmdd_not_held": _s5a,
    "s5b_no_date_kst_today": _s5b,
}


def names() -> list[str]:
    return list(_BUILDERS)


def get(name: str) -> dict[str, Any]:
    """A private deep copy of a ready scenario."""
    try:
        return copy.deepcopy(_BUILDERS[name]())
    except KeyError:
        raise KeyError(f"unknown scenario {name!r}; known: {names()}") from None
