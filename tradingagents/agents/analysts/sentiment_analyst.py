"""Sentiment analyst: one sentiment report from three sources.

The node fetches its sources before calling the model and puts them in the
prompt, so the model reports on data it was given rather than inventing posts:

  1. News headlines: Yahoo Finance
  2. StockTwits messages: the cashtag stream, with Bullish/Bearish tags
  3. Reddit posts: r/wallstreetbets, r/stocks, r/investing

Each source is trimmed to the analysis window. With a TypeSafe key, the social
posts are screened by Jev first (see post_screen). These feeds serve recent items
and are not archived, so a historical run's sentiment inputs are not
point-in-time.

The report is a SentimentReport through structured output where the provider
supports it and free text otherwise, so the band, score and confidence header
reads the same across providers.
"""

import logging
from datetime import datetime, timedelta

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.context import (
    get_instrument_context_from_state,
    get_language_instruction,
    resolve_instrument_identity,
)
from tradingagents.agents.post_screen import jev_screen
from tradingagents.agents.schemas import SentimentReport, render_sentiment_report
from tradingagents.agents.structured import (
    NO_EXTERNAL_TOOLS,
    bind_structured,
    invoke_structured_or_freetext,
)
from tradingagents.agents.tools import get_news
from tradingagents.dataflows.vendors.reddit import fetch_reddit_posts
from tradingagents.dataflows.vendors.stocktwits import fetch_stocktwits_messages

logger = logging.getLogger(__name__)


def _seven_days_back(trade_date: str) -> str:
    return (datetime.strptime(trade_date, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")


def _maybe_fetch_kr_discussion(ticker: str) -> str:
    """Fetch Naver 종목토론방 retail sentiment for KR tickers, opt-in only.

    Returns "" (omitted from the prompt) unless the ticker is Korean AND
    config['enable_kr_discussion_sentiment'] is true. Off by default because it
    queries an undocumented Naver endpoint; author identities are never
    included (see naver_discussion).
    """
    try:
        from tradingagents.dataflows.config import get_config
        from tradingagents.dataflows.kr_utils import is_kr_ticker
        if not (is_kr_ticker(ticker) and get_config().get("enable_kr_discussion_sentiment")):
            return ""
        from tradingagents.dataflows.naver_discussion import fetch_discussion_sentiment
        return fetch_discussion_sentiment(ticker)
    except Exception as exc:  # noqa: BLE001 — never block the node
        logger.warning("KR discussion sentiment skipped for %s: %s", ticker, exc)
        return ""


def _resolve_company_name(ticker: str) -> str | None:
    """Best-effort lookup of a ticker's company name.

    Used by the Reddit fetcher to expand recall: short tickers (NTRA,
    PSNL, BWXT) rarely appear verbatim in retail posts; the company name
    catches the same threads under their conversational spelling.

    Read from the run's instrument identity -- the lru_cached lookup resolved
    once at run start, so this is a cache hit mid-run, and the same name the
    Jev screen uses. Agents never import yfinance themselves; the data layer
    owns the vendor.

    Returns ``None`` on any failure — the Reddit fetcher falls back to a
    ticker-only search, matching the pre-existing behaviour.
    """
    try:
        name = resolve_instrument_identity(ticker).get("company_name")
    except Exception as exc:  # noqa: BLE001 — a recall aid must never block the node
        logger.warning("Could not resolve company name for %s: %s", ticker, exc)
        return None
    return name if isinstance(name, str) and name.strip() else None


def create_sentiment_analyst(llm):
    """Create a sentiment analyst node for the trading graph.

    Pre-fetches news + StockTwits + Reddit data, injects them into the
    prompt as structured blocks, and produces a deterministic sentiment
    report via structured output (with a free-text fallback for providers
    that do not support it).
    """
    structured_llm = bind_structured(llm, SentimentReport, "Sentiment Analyst")

    def sentiment_analyst_node(state):
        ticker = state["company_of_interest"]
        end_date = state["trade_date"]
        start_date = _seven_days_back(end_date)
        instrument_context = get_instrument_context_from_state(state)

        # Pre-fetch all three sources. Each fetcher degrades gracefully and
        # returns a string (no exceptions surface from here), so the LLM
        # always sees something — either real data or a clear placeholder.
        news_block = get_news.func(ticker, start_date, end_date)
        # Pass the analysis window so a historical run trims social posts to it
        # instead of leaking today's chatter into a backtest (#1220).
        screen = jev_screen(ticker)
        stocktwits_block = fetch_stocktwits_messages(
            ticker, limit=30, start_date=start_date, end_date=end_date, screen=screen
        )
        # Reddit users typically spell out the company name rather than the
        # ticker (esp. for short, non-obvious tickers like NTRA/PSNL/BWXT),
        # so we OR the ticker with the resolved company name to widen recall.
        reddit_block = fetch_reddit_posts(
            ticker,
            start_date=start_date,
            end_date=end_date,
            screen=screen,
            company_name=_resolve_company_name(ticker),
        )

        # Korean retail sentiment from Naver 종목토론방 — opt-in (off by default),
        # KR tickers only. StockTwits/Reddit structurally miss Korean retail.
        # ⚠️ 이 소스만 창 트리밍이 없다 — 네이버 응답에 신뢰할 만한 타임스탬프가
        # 없어 #1220 의 look-ahead 가드를 적용할 수 없다. 과거 날짜 분석에서는
        # KR 종목토론방만 '현재' 잡담을 볼 수 있다.
        kr_discussion_block = _maybe_fetch_kr_discussion(ticker)

        system_message = _build_system_message(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            news_block=news_block,
            stocktwits_block=stocktwits_block,
            reddit_block=reddit_block,
            kr_discussion_block=kr_discussion_block,
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Report what your tools support; another agent decides the trade."
                    # No tool-calling here: the data is pre-fetched into the
                    # prompt, so tool-range wording would only invite a
                    # hallucinated tool call (#1130).
                    " Today's date is {current_date}; treat it as 'now' for all analysis. {instrument_context}"
                    " " + NO_EXTERNAL_TOOLS +
                    "\n{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(current_date=end_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        # Format the template into a concrete message list so the structured
        # and free-text paths receive the same input. No bind_tools — the
        # data is already in the prompt.
        formatted_messages = prompt.format_messages(messages=state["messages"])

        report_text, _ = invoke_structured_or_freetext(
            structured_llm,
            llm,
            formatted_messages,
            render_sentiment_report,
            "Sentiment Analyst",
        )

        return {
            "messages": [AIMessage(content=report_text)],
            "sentiment_report": report_text,
        }

    return sentiment_analyst_node


def _build_system_message(
    *,
    ticker: str,
    start_date: str,
    end_date: str,
    news_block: str,
    stocktwits_block: str,
    reddit_block: str,
    kr_discussion_block: str = "",
) -> str:
    """Assemble the sentiment-analyst system message with structured data blocks."""
    # Korean 종목토론방 section is only present for KR tickers with the opt-in
    # flag enabled (otherwise kr_discussion_block is "" and the section is omitted).
    kr_section = ""
    if kr_discussion_block:
        kr_section = f"""

### Naver 종목토론방 — Korean retail discussion board (KR tickers)
Korean retail chatter that StockTwits/English-Reddit miss. Read the Korean text
for mood; 추천/비추천 are community reactions to a post, not a bull/bear label.

<start_of_kr_discussion>
{kr_discussion_block}
<end_of_kr_discussion>"""

    return f"""You are a financial market sentiment analyst. Your task is to produce a comprehensive sentiment report for {ticker} covering the period from {start_date} to {end_date}, drawing on the complementary data sources that have already been collected for you.

## Data sources (pre-fetched, in this prompt)

### News headlines — Yahoo Finance, past 7 days
Institutional framing. Fact-driven, slower-moving signal.

<start_of_news>
{news_block}
<end_of_news>

### StockTwits messages — retail-trader social platform indexed by cashtag
Fast-moving signal. Each message carries a user-labeled sentiment tag (Bullish / Bearish / no-label) plus the message body.

<start_of_stocktwits>
{stocktwits_block}
<end_of_stocktwits>

### Reddit posts — r/wallstreetbets, r/stocks, r/investing (past 7 days)
Community discussion, without vote or comment counts. Subreddit character matters (r/wallstreetbets is often contrarian/exuberant; r/stocks more measured; r/investing longer-term).

<start_of_reddit>
{reddit_block}
<end_of_reddit>{kr_section}

## How to analyze this data (best practices)

1. **Read the StockTwits Bullish/Bearish ratio as a leading retail-sentiment signal.** A 70/30 bullish/bearish split is moderately bullish; ≥90/10 may indicate over-extension and contrarian risk; 50/50 is uncertainty. Sample size matters — base rates on the actual message count, not percentages alone. A block headed "Screened by Jev" has had off-topic posts removed; its stance count is a classifier's read of every on-topic post fetched, labelled or not, of which the posts listed are a sample. Read it alongside the user tags.

2. **Look for cross-source divergences.** If news framing is bearish but StockTwits is overwhelmingly bullish, that mismatch is itself a signal — it can mean retail is leaning into a thesis the news flow hasn't caught up to (or vice versa, that retail is chasing while institutions are cautious).

3. **Read Reddit posts for substance.** The feed carries no vote or comment counts, so judge a post by its body excerpt, not its title alone, and do not infer engagement.

4. **Distinguish opinion from event.** A news headline ("Nvidia announces $500M Corning deal") is an event; a StockTwits post ("buying NVDA, this is going to moon") is opinion. Both are inputs but should be weighted differently in your conclusions.

5. **Identify recurring narrative themes.** What topic keeps coming up across sources? That's the dominant narrative driving current sentiment.

6. **Be honest about data limits.** If StockTwits returned only a handful of messages, or one or more sources returned an "<unavailable>" placeholder, the sentiment read is less robust — flag this explicitly in the `confidence` field and the narrative. If the sources are silent on a given subreddit, say so.

7. **Identify catalysts and risks** that emerge across sources — news of upcoming earnings, product launches, competitive threats, macro headlines, etc.

8. **Past sentiment is not predictive.** Frame your conclusions as signal for the trader to weigh alongside fundamentals and technicals, not as a price call.

## Output fields

Fill the following fields:

- **overall_band**: Exactly one of Bullish / Mildly Bullish / Neutral / Mixed / Mildly Bearish / Bearish. Use Mixed when sources point in clearly different directions; Neutral only when all sources are genuinely silent.
- **overall_score**: A number from 0 (maximally bearish) to 10 (maximally bullish); 5 is neutral. Keep it consistent with overall_band.
- **confidence**: low / medium / high, based on data quality and sample size.
- **narrative**: Full source-by-source breakdown, divergences, dominant narrative themes, catalysts and risks, and a markdown summary table of key sentiment signals (direction, source, supporting evidence).

{get_language_instruction()}"""
