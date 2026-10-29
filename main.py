"""Programmatic entry point.

Usage:
    python main.py TICKER [DATE]

TICKER is required (e.g. NVDA, MU, 005930.KS). DATE is optional (YYYY-MM-DD or
YYYYMMDD) and defaults to today in the process's time zone (TZ). 아래 실행 설정은
Anthropic API 를 직접 부르는(ANTHROPIC_API_KEY) 계층형 Claude 실행이다 —
``build_config`` 를 고치거나 주석 처리하면 DEFAULT_CONFIG 로 돌아간다.
"""

import argparse
import datetime
from pathlib import Path

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph


def _iso_date(value: str) -> str:
    """argparse type: a calendar date, returned as YYYY-MM-DD, so a bad date fails fast.

    Accepts YYYY-MM-DD and the basic form YYYYMMDD, which a nightly caller passes
    (a KST date). ``date.fromisoformat`` reads the basic form only from Python
    3.11 on, so it is parsed explicitly: on 3.10 the same argv would otherwise
    fail before the run starts.
    """
    try:
        if len(value) == 8 and value.isascii() and value.isdigit():
            return datetime.datetime.strptime(value, "%Y%m%d").date().isoformat()
        return datetime.date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"date must be YYYY-MM-DD or YYYYMMDD, got {value!r}"
        ) from exc


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a TradingAgents analysis.")
    parser.add_argument("ticker", help="Ticker to analyze (required), e.g. NVDA, MU, 005930.KS")
    parser.add_argument(
        "date",
        nargs="?",
        type=_iso_date,
        default=datetime.date.today().isoformat(),
        help="Analysis date, YYYY-MM-DD or YYYYMMDD (default: today in the local time zone)",
    )
    return parser.parse_args(argv)


def build_config() -> dict:
    """DEFAULT_CONFIG 에 Anthropic API 직접 호출의 계층형 Claude 실행을 더한다.

    필요한 것은 ANTHROPIC_API_KEY(환경 변수 또는 .env) 하나다: 공급자 ``anthropic`` 은
    langchain-anthropic 의 ChatAnthropic(기본 의존성)을 만들어 api.anthropic.com 을
    직접 부른다 — GCP 프로젝트도, ADC 도, [vertex] extra 도 필요 없다. 두 심판
    (Research/Portfolio Manager)은 Opus 5.5 effort xhigh, 나머지 역할은 모두 Sonnet 5.5
    effort high 다. thinking/max_tokens 는 공통이다. 구조화 호출(Research Manager,
    Trader, Portfolio Manager)은 스키마를 유일한 tool 로 tool_choice "auto" 로 묶는다:
    5.5 모델은 강제 tool 호출을 400 으로 거절한다. update 를 주석 처리하면
    DEFAULT_CONFIG(OpenAI + .env)로 돌아간다.

    thinking 은 max_tokens 에 포함된다. 같은 모델을 Vertex 에서 잰 값(2026-10-01, KR
    실행, 입력 1.2~2만 토큰): Opus 5.5 effort max 는 모든 심판 호출에서 20000 을 다
    thinking 에 썼다(tool 호출 없음 -> TRADE_PLAN_JSON 없음). xhigh 에서 심판은 1.1~1.3만,
    Sonnet 5.5 bear researcher(high)는 1.94만까지 썼다. 32000 은 둘 다에 여유가 있다.
    ChatAnthropic 은 SDK 클라이언트를 timeout=None(default_request_timeout)으로 만들어,
    SDK 의 비스트리밍 "Streaming is required" 가드(SDK 기본 timeout 에서 ~21.3k)가 여기에는
    걸리지 않는다: 32000 토큰 비스트리밍 요청이 그대로 나간다
    (tests/test_anthropic_direct.py 가 고정).
    """
    config = DEFAULT_CONFIG.copy()
    config.update({
        "llm_provider": "anthropic",         # 직접 API(ANTHROPIC_API_KEY), Vertex 아님
        # SDK 자신의 엔드포인트(api.anthropic.com)로 고정한다. DEFAULT_CONFIG 의 backend_url 은
        # TRADINGAGENTS_LLM_BACKEND_URL(환경 변수나 cwd 와 그 위의 .env)에서 올 수 있고,
        # 직접 anthropic 공급자는 그것을 ChatAnthropic(base_url=...) 으로 써서 모든 요청과
        # API 키를 그 엔드포인트로 보낸다(vertex_* 공급자는 그 값을 쓰지 않았다).
        "backend_url": None,
        "deep_think_llm": "claude-opus-5-5",
        "quick_think_llm": "claude-sonnet-5-5",
        "anthropic_thinking": "adaptive",
        "anthropic_max_tokens": 32000,
        "anthropic_effort": "high",          # quick tier (Sonnet 5.5) default
        "output_language": "Korean",         # localize the user-facing report/decision
        # KR-only vendors, opt-in by design (default_config keeps them off).
        # Both raise for non-KR tickers so the chain falls through to yfinance
        # unchanged — US runs are unaffected. wisereport's audited-actuals half
        # needs DART_API_KEY and degrades to yfinance without it.
        "data_vendors": {
            **DEFAULT_CONFIG["data_vendors"],
            "news_data": "naver,yfinance",
            "fundamental_data": "wisereport,yfinance",
        },
        # Naver 종목토론방 retail sentiment (KR tickers only; one HTTP GET,
        # degrades to a placeholder on failure — never blocks the node).
        "enable_kr_discussion_sentiment": True,
        # This runner never passes a portfolio: the account context reaches the
        # Portfolio Manager only, through TRADINGAGENTS_POSITION_CONTEXT. Without
        # this, every Trader / risk-debater prompt (and the PM's, when no position
        # is injected) would gain upstream's "Portfolio context: not provided"
        # notice that the consumer's prompts never carried.
        "portfolio_notice_when_absent": False,
        # Read that variable into the run (the interactive CLI and the backtest
        # switch the read off). Pinned here so the consumer's account channel
        # does not hang on DEFAULT_CONFIG's default.
        "position_context_from_env": True,
        "role_models": {                     # override the two deep judges -> Opus / xhigh
            "research_manager": {
                "provider": "anthropic", "model": "claude-opus-5-5",
                "anthropic_effort": "xhigh",
            },
            "portfolio_manager": {
                "provider": "anthropic", "model": "claude-opus-5-5",
                "anthropic_effort": "xhigh",
            },
        },
    })
    return config


def main(argv=None) -> None:
    args = parse_args(argv)
    config = build_config()
    # selected_analysts defaults to all four; listed explicitly so the run scope
    # is obvious and easy to trim.
    ta = TradingAgentsGraph(
        selected_analysts=("market", "social", "news", "fundamentals"),
        debug=True,
        config=config,
    )
    final_state, decision = ta.propagate(args.ticker, args.date)
    plan_obj = final_state.get("portfolio_decision_obj")
    # The decision line is the Portfolio Manager's typed rating whenever its
    # structured call succeeded -- the same value as TRADE_PLAN_JSON.rating below.
    # Re-reading it from the rendered markdown could pick up another rating the
    # prose quotes (a founding rating, a revision note). ``.value`` keeps the bare
    # word: an Enum formats as 'PortfolioRating.SELL' on Python 3.11. Without a
    # typed decision (free-text fallback) it is the parsed signal, REVIEW when no
    # rating can be read.
    print(plan_obj.rating.value if plan_obj is not None else decision)
    # Machine-readable trade plan for downstream callers that run this as a
    # subprocess and parse stdout. The known consumer is AlphaPulse
    # (alphapulse/webapp/services/trade_plan.py), which turns this into order
    # drafts. Renaming a field or dropping this line raises nothing on either
    # side -- the consumer stores plan_status='unparsed' and the UI just says
    # "no plan" -- so change it only together with that parser and its fixture.
    # Absent when the Portfolio Manager's
    # structured call fell back to free text -- "no plan" is a valid state and
    # the consumer must not invent one. executive_summary/investment_thesis are
    # already in the saved report, so they are excluded to keep the line small.
    if plan_obj is not None:
        print("TRADE_PLAN_JSON: " + plan_obj.model_dump_json(
            exclude={"executive_summary", "investment_thesis"},
        ))
    # Write the report tree: per-section 1_analysts..5_portfolio markdown plus a
    # consolidated complete_report.md, through the writer the CLI uses, with the
    # CLI's rich header (the resolved company label and a per-role model table
    # rendered from `config`). Imported only here, after the run; main.py does
    # not import cli.main (the interactive app).
    from cli.report_meta import build_report_header
    from tradingagents.dataflows.symbols import safe_ticker_component
    from tradingagents.reporting import write_report_tree
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = Path(config["results_dir"]) / "reports" / f"{safe_ticker_component(args.ticker)}_{stamp}"
    report_path = write_report_tree(
        final_state, args.ticker, save_path, header=build_report_header(args.ticker, config),
    )
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
