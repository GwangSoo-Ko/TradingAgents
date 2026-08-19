"""매도(축소) 계획을 담는 스키마 필드.

매도 계획은 지금까지 매수와 같은 필드를 빌려 썼다 — 목표 비중(total_weight_pct)에
잔여 비중을 넣고, OCO(둘 중 먼저 오는 것)를 트랜치 두 개로 쪼개고, 지표값(10EMA)을
주문 밴드(price_high)에 넣었다. 셋 다 숫자가 주문으로 바뀌는 자리라 위험하다.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from tradingagents.agents.schemas import (
    ExitTarget,
    KillSwitch,
    PortfolioDecision,
    Tranche,
    TrancheTrigger,
    render_pm_decision,
)

pytestmark = pytest.mark.unit


def test_tranche_trigger_carries_trail_pct_for_trailing_stop():
    """트레일링 스톱은 가격 밴드가 아니라 비율이다 — 담을 필드가 있어야 한다."""
    t = TrancheTrigger(kind="trailing", trail_pct=8.0)
    assert t.trail_pct == 8.0
    assert t.price is None


def test_tranche_holds_multiple_triggers_as_oco():
    """한 트랜치에 트리거 여럿 = 먼저 오는 것이 이긴다(OCO). pct 는 한 번만 센다."""
    tr = Tranche(
        seq=1,
        pct=35.0,
        trigger="conditional",
        triggers=[
            TrancheTrigger(kind="take_profit", price=11500.0),
            TrancheTrigger(kind="stop", price=10600.0),
        ],
    )
    assert len(tr.triggers) == 2
    assert tr.pct == 35.0


def test_reference_price_is_separate_from_execution_band():
    """지표값(10EMA)은 주문 밴드가 아니다 — 별도 필드에 담긴다."""
    t = TrancheTrigger(
        kind="stop", reference_price=14628.0, reference_label="10EMA",
        condition="종가 기준 2거래일 연속 10EMA 하회",
    )
    assert t.reference_price == 14628.0
    assert t.price is None


def test_exit_target_supports_cost_recovery():
    assert ExitTarget(kind="cost_recovery").remaining_weight_pct is None
    assert ExitTarget(kind="weight", remaining_weight_pct=1.5).remaining_weight_pct == 1.5


def test_portfolio_decision_defaults_keep_backward_compatibility():
    """새 필드는 전부 optional — 기존 호출부가 깨지지 않는다."""
    d = PortfolioDecision(
        rating="Hold", executive_summary="s", investment_thesis="t",
    )
    assert d.exit_target is None
    assert d.kill_switch is None
    assert d.tranches == []


def test_kill_switch_requires_condition():
    """조건 없는 킬스위치는 가격 하나만 남아 왜 청산하는지 아무도 모른다."""
    with pytest.raises(ValidationError):
        KillSwitch(price=9100.0)

    ks = KillSwitch(price=9100.0, condition="200SMA 훼손 시 잔여 전량 청산")
    assert ks.condition


def test_kill_switch_may_be_condition_only():
    """가격 없는 조건(공시·일정)도 킬스위치다 — 가격을 필수로 만들면 그걸 못 담는다."""
    assert KillSwitch(condition="10월 임시주총 연기 공시 시 전량 청산").price is None


def test_render_shows_triggers_under_their_tranche():
    """자유텍스트 폴백이 아니어도 사람이 읽는 기록은 마크다운뿐이다 — 조건이 여기서
    빠지면 5_portfolio/decision.md 와 메모리 로그에 매도 조건이 사라진다."""
    md = render_pm_decision(PortfolioDecision(
        rating="Underweight", executive_summary="s", investment_thesis="t",
        tranches=[
            Tranche(seq=1, pct=40, price_low=10800, price_high=11000, trigger="immediate"),
            Tranche(
                seq=2, pct=40, trigger="conditional",
                triggers=[
                    TrancheTrigger(kind="take_profit", price=11500.0),
                    TrancheTrigger(kind="trailing", trail_pct=8.0),
                    TrancheTrigger(kind="stop", reference_price=14628.0,
                                   reference_label="10EMA",
                                   condition="종가 기준 2거래일 연속 10EMA 하회"),
                ],
            ),
            Tranche(
                seq=3, pct=20, trigger="conditional",
                triggers=[TrancheTrigger(kind="event", condition="10월 임시주총")],
            ),
        ],
    ))

    assert "take_profit @11500" in md
    assert "trailing -8%" in md
    assert "10EMA=14628" in md
    assert "종가 기준 2거래일 연속 10EMA 하회" in md
    # Each trigger sits under ITS OWN tranche. Asserting only "#2 precedes its own
    # trigger" is trivially true even if the loop is hoisted out of the tranche
    # loop and every trigger is dumped after every tranche line -- so anchor on a
    # LATER tranche's header instead: #2's triggers must come before "- #3 ".
    assert md.index("- #2 ") < md.index("take_profit @11500") < md.index("- #3 ")
    assert md.index("- #3 ") < md.index("event 10월 임시주총")


def test_render_shows_exit_target_and_kill_switch_after_the_existing_headers():
    """신규 블록은 기존 헤더 뒤에 온다 — 다운스트림은 순서를 전제로 읽는다."""
    md = render_pm_decision(PortfolioDecision(
        rating="Underweight", executive_summary="s", investment_thesis="t",
        tranches=[Tranche(seq=1, pct=100, trigger="immediate")],
        exit_target=ExitTarget(kind="weight", remaining_weight_pct=1.5),
        kill_switch=KillSwitch(price=9100.0, condition="200SMA 훼손 시 잔여 전량 청산"),
    ))

    assert "**Exit Target**: weight (1.5%)" in md
    assert "**Kill Switch** @ 9100: 200SMA 훼손 시 잔여 전량 청산" in md
    order = ["**Rating**", "**Executive Summary**", "**Investment Thesis**",
             "**Entry Plan**", "**Exit Target**", "**Kill Switch**"]
    positions = [md.index(h) for h in order]
    assert positions == sorted(positions), dict(zip(order, positions, strict=True))


def test_render_exit_target_omits_the_weight_detail_when_not_a_weight_target():
    """cost_recovery/full 에는 잔여 비중이 없다 — 없는 숫자를 지어내면 안 된다."""
    md = render_pm_decision(PortfolioDecision(
        rating="Sell", executive_summary="s", investment_thesis="t",
        exit_target=ExitTarget(kind="cost_recovery"),
    ))

    assert "**Exit Target**: cost_recovery" in md
    assert "%" not in md.split("**Exit Target**")[1]


def test_render_never_prints_a_residual_weight_on_a_full_exit():
    """kind 를 안 보고 값만 보면, 전량 청산 계획이 '1.5% 남긴다'로 렌더된다.

    kind='full' + remaining_weight_pct=1.5 는 구성 가능한 조합이다(스키마가 막지
    않는다). cost_recovery 케이스만으로는 이 결함이 안 잡힌다 — 거기선 값이 애초에
    None 이라 `kind == 'weight'` 조건을 지워도 테스트가 초록으로 남는다.
    """
    md = render_pm_decision(PortfolioDecision(
        rating="Sell", executive_summary="s", investment_thesis="t",
        exit_target=ExitTarget(kind="full", remaining_weight_pct=1.5),
    ))

    assert "**Exit Target**: full" in md
    assert "1.5" not in md.split("**Exit Target**")[1]


def test_render_omits_the_new_blocks_when_absent():
    """매수 계획에는 매도 전용 블록이 없다 — 빈 헤더가 새면 리포트가 거짓말을 한다."""
    md = render_pm_decision(PortfolioDecision(
        rating="Buy", executive_summary="s", investment_thesis="t",
    ))

    assert "**Exit Target**" not in md
    assert "**Kill Switch**" not in md
