"""Programmatic entry point.

Usage:
    python main.py TICKER [DATE]

TICKER is required (e.g. NVDA, MU, 005930.KS). DATE is optional (YYYY-MM-DD) and
defaults to today. The run config below is a tiered Vertex Model Garden run on
Claude — edit or comment out ``build_config`` to fall back to DEFAULT_CONFIG.
"""

import argparse
import datetime
from pathlib import Path

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph


def _iso_date(value: str) -> str:
    """argparse type: accept only YYYY-MM-DD so a bad date fails fast."""
    try:
        return datetime.date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"date must be YYYY-MM-DD, got {value!r}") from exc


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a TradingAgents analysis.")
    parser.add_argument("ticker", help="Ticker to analyze (required), e.g. NVDA, MU, 005930.KS")
    parser.add_argument(
        "date",
        nargs="?",
        type=_iso_date,
        default=datetime.date.today().isoformat(),
        help="Analysis date YYYY-MM-DD (default: today)",
    )
    return parser.parse_args(argv)


def build_config() -> dict:
    """DEFAULT_CONFIG plus a tiered Vertex Claude run.

    No vendor API key — ADC auth + the optional [vertex] extra; needs
    GOOGLE_CLOUD_PROJECT and ``gcloud auth application-default login``. The two
    deep judges (Research/Portfolio Manager) run Opus 4.8 at max effort; every
    other role runs Sonnet 5 at high effort. thinking/max_tokens are shared;
    vertex_project/location resolve from GOOGLE_CLOUD_PROJECT / GOOGLE_CLOUD_LOCATION
    (default "global"). max_tokens stays <= ~21.3k so the non-streaming node calls
    don't trip the Anthropic SDK's "streaming required" guard. Comment the update
    out to fall back to DEFAULT_CONFIG (OpenAI + .env).
    """
    config = DEFAULT_CONFIG.copy()
    config.update({
        "llm_provider": "vertex_anthropic",
        "deep_think_llm": "claude-opus-5",
        "quick_think_llm": "claude-sonnet-5",
        "anthropic_thinking": "adaptive",
        "anthropic_max_tokens": 20000,
        "anthropic_effort": "high",          # quick tier (Sonnet 5) default
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
        "role_models": {                     # override the two deep judges -> Opus / max
            "research_manager": {
                "provider": "vertex_anthropic", "model": "claude-opus-5",
                "anthropic_effort": "max",
            },
            "portfolio_manager": {
                "provider": "vertex_anthropic", "model": "claude-opus-5",
                "anthropic_effort": "max",
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
    print(decision)
    # Write the CLI's rich-header report tree: per-section 1_analysts..5_portfolio
    # markdown plus a consolidated complete_report.md whose header carries the
    # resolved company label and a per-role model table. Reuse the CLI writer so
    # the on-disk output matches `tradingagents` exactly (needs the `config`).
    from cli.main import save_report_to_disk
    from tradingagents.dataflows.utils import safe_ticker_component
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = Path(config["results_dir"]) / "reports" / f"{safe_ticker_component(args.ticker)}_{stamp}"
    report_path = save_report_to_disk(final_state, args.ticker, save_path, config)
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
