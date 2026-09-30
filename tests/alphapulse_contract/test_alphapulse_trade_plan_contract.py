"""TRADE_PLAN_JSON schema contract: what alpha-pulse reads from the Portfolio Manager's plan.

alpha-pulse turns the LAST ``TRADE_PLAN_JSON: {...}`` stdout line of main.py into order drafts
(its trade-plan module, named in main.py's comment at the print). main.py prints
``PortfolioDecision.model_dump_json(exclude={"executive_summary", "investment_thesis"})`` of the
Portfolio Manager's typed decision. When the model's plan does not validate, the fork's
structured call falls back to free text and main.py prints NO plan line: alpha-pulse stores
plan_status 'unparsed' and drafts nothing. That fail-closed behaviour IS the contract. Downstream a
null plan number is not "unknown": a missing immediate band means an order at the current price
with the band gate off, a missing stop means no stop, and a separator-stripped number is simply a
different number.

Frozen here, in-process on THIS checkout's schema:
  * the fields alpha-pulse reads keep their JSON type, nullability, required-ness and closed
    vocabularies (unrelated new optional fields are allowed);
  * reviewed plans (s1's synthetic Sell plan among them) survive validation plus main.py's dump
    call value-for-value, and the line stays one line of strict JSON;
  * the plan shape of before the 2026-08 trigger change is rejected whole, not reinterpreted;
  * non-finite numbers never reach the line as NaN/Infinity tokens;
  * an unreadable plan number fails the WHOLE plan, a placeholder means "not provided", a plain
    numeric string parses.
And end to end through the real main.py (harness subprocess): a readable plan is printed exactly as
the schema dumps it, and an unreadable stop / trigger price yields no TRADE_PLAN_JSON line while
the run still exits 0 with its report.

The printed line is read with ``contract_reader`` (the documented one-line grammar: one strict
JSON object, NaN/Infinity rejected, anything unreadable = no plan).

Sub-models (Tranche, TrancheTrigger, ExitTarget, KillSwitch, PlanRevision) are reached through
PortfolioDecision's own field annotations, i.e. exactly the classes the plan uses, whatever a merge
renames them to. Test ids keep today's ``Model.field`` names for readability.
"""

from __future__ import annotations

import copy
import enum
import inspect
import math
import types
import typing
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pydantic
import pytest

from . import contract_reader, scenarios
from ._compat import ABSENT, canonical_json, get_symbol, project_to_expected
from .harness import DECISION_WORDS, REPO_ROOT, REPORT_SAVED_LINE_RE

S1 = "s1_nightly_kr_holding_sell"

# Where the Portfolio Manager's schema lives: a2981a7 and upstream v0.5.1/v0.5.2 all keep it here.
# If a merge moves it, put the new home FIRST; get_symbol fails loudly when none resolves.
SCHEMA_MODULES = ("tradingagents.agents.schemas",)

PLAN_PREFIX = "TRADE_PLAN_JSON: "

# ---------------------------------------------------------------------------------------------
# What alpha-pulse reads from the line. Hand-written from the plan contract the fork documents
# (docs/INTEGRATION.md §1b "The plan contract") and its PortfolioDecision schema at a2981a7.
# ---------------------------------------------------------------------------------------------

RATINGS = ("Buy", "Overweight", "Hold", "Underweight", "Sell")
TRANCHE_TRIGGERS = ("immediate", "conditional")
TRIGGER_KINDS = ("take_profit", "stop", "trailing", "event")
EXIT_TARGET_KINDS = ("weight", "cost_recovery", "full")
REVISION_KINDS = ("new_information", "price_action", "thesis_error", "tactical")


@dataclass(frozen=True)
class Slot:
    """One key alpha-pulse reads: its JSON type and whether the model may leave it out/null."""

    kind: str  # "number" | "integer" | "string" | "enum" | "array" | "object"
    required: bool = False  # the model must always supply it (no default)
    nullable: bool | None = True  # None: alpha-pulse reads null and empty alike (lists)
    values: tuple[str, ...] = ()  # the closed vocabulary when kind == "enum"
    fields: Mapping[str, Slot] | None = None  # nested object / array-item contract


TRIGGER_CONTRACT = {
    "kind": Slot("enum", required=True, nullable=False, values=TRIGGER_KINDS),
    "price": Slot("number"),
    "trail_pct": Slot("number"),
    "reference_price": Slot("number"),
    "reference_label": Slot("string"),
    "condition": Slot("string"),
}
TRANCHE_CONTRACT = {
    "seq": Slot("integer", required=True, nullable=False),
    "pct": Slot("number", required=True, nullable=False),
    "price_low": Slot("number"),
    "price_high": Slot("number"),
    "trigger": Slot("enum", required=True, nullable=False, values=TRANCHE_TRIGGERS),
    "triggers": Slot("array", nullable=None, fields=TRIGGER_CONTRACT),
    "condition": Slot("string"),
}
EXIT_TARGET_CONTRACT = {
    "kind": Slot("enum", required=True, nullable=False, values=EXIT_TARGET_KINDS),
    "remaining_weight_pct": Slot("number"),
}
KILL_SWITCH_CONTRACT = {
    "price": Slot("number"),
    "condition": Slot("string", required=True, nullable=False),
}
REVISION_CONTRACT = {
    "kind": Slot("enum", required=True, nullable=False, values=REVISION_KINDS),
    "note": Slot("string", required=True, nullable=False),
}
PLAN_CONTRACT = {
    "rating": Slot("enum", required=True, nullable=False, values=RATINGS),
    "price_target": Slot("number"),
    "time_horizon": Slot("string"),
    "total_weight_pct": Slot("number"),
    "stop_loss": Slot("number"),
    "tranches": Slot("array", nullable=None, fields=TRANCHE_CONTRACT),
    "exit_target": Slot("object", fields=EXIT_TARGET_CONTRACT),
    "kill_switch": Slot("object", fields=KILL_SWITCH_CONTRACT),
    "revision": Slot("object", fields=REVISION_CONTRACT),
}
# Required by the schema but not on the line (main.py excludes them; the saved report shows them).
PROSE_FIELDS = frozenset({"executive_summary", "investment_thesis"})

# ---------------------------------------------------------------------------------------------
# Plans the Portfolio Manager sends (raw structured-output tool args)
# ---------------------------------------------------------------------------------------------

# A hand-written reduction plan that covers what the Sell plan does not: a 'weight' exit target,
# a kill switch without a price, a conditional tranche without a band, a price_action revision
# and prose with a line break (and the trailing trigger the trail_pct number slot uses). It
# restates, in today's shape, the synthetic old-shape Underweight plan of
# fixtures/synthetic_legacy_underweight_plan.json (target 8,420, stop 8,150, band 8,980~9,120,
# 2% remaining weight).
UNDERWEIGHT_PM: dict[str, Any] = {
    "rating": "Underweight",
    "executive_summary": "보유분을 두 단계로 줄여 순자산 대비 2%만 남긴다.",
    "investment_thesis": "영업이익률 개선이 멈췄고 다음 분기 이익 전망이 낮아지고 있다.",
    "price_target": 8420.0,
    "time_horizon": None,
    "total_weight_pct": None,
    "stop_loss": 8150.0,
    "tranches": [
        {"seq": 1, "pct": 60.0, "price_low": 8980.0, "price_high": 9120.0,
         "trigger": "immediate", "triggers": [], "condition": None},
        {"seq": 2, "pct": 40.0, "price_low": None, "price_high": None, "trigger": "conditional",
         "triggers": [{"kind": "trailing", "price": None, "trail_pct": 8.0,
                       "reference_price": None, "reference_label": None,
                       "condition": "고점 대비 8% 하락 시 잔여 축소분 집행"}],
         "condition": "1차 집행 뒤 추세가 꺾이면.\n실적 발표 전에는 보류."},
    ],
    "exit_target": {"kind": "weight", "remaining_weight_pct": 2.0},
    "kill_switch": {"price": None,
                    "condition": "다음 분기 영업이익률이 다시 떨어졌다는 공시가 나오면 전량 정리"},
    "revision": {"kind": "price_action",
                 "note": "20일선 8,690원을 이틀 연속 하회해 진입 당시의 추세 논거가 깨졌다."},
}

BASE_PLANS: dict[str, Callable[[], dict[str, Any]]] = {
    # s1: the synthetic Sell plan (fixtures/synthetic_sell_plan.json) + prose + a revision.
    "sell": scenarios.reit_sell_pm,
    # s5: a fresh-entry Buy with a target weight.
    "buy": lambda: copy.deepcopy(scenarios.SAMSUNG_BUY_PM),
    "underweight": lambda: copy.deepcopy(UNDERWEIGHT_PM),
    # A rating with nothing else: every optional part left out.
    "minimal": lambda: {"rating": "Hold", "executive_summary": "관망한다.",
                        "investment_thesis": "양쪽 근거가 균형을 이룬다."},
}

_TRIGGER_OF_KIND = {
    "take_profit": {"kind": "take_profit", "price": 11060.0, "trail_pct": None,
                    "reference_price": None, "reference_label": None,
                    "condition": "11,060원 도달 시 체결"},
    "stop": {"kind": "stop", "price": 10380.0, "trail_pct": None, "reference_price": 10405.0,
             "reference_label": "60SMA", "condition": "종가 기준 60일선 하향 이탈"},
    "trailing": {"kind": "trailing", "price": None, "trail_pct": 8.0, "reference_price": None,
                 "reference_label": None, "condition": "고점 대비 8% 하락"},
    "event": {"kind": "event", "price": None, "trail_pct": None, "reference_price": None,
              "reference_label": None, "condition": "정기 공시 발표"},
}
_EXIT_TARGET_OF_KIND = {
    "weight": ("underweight", {"kind": "weight", "remaining_weight_pct": 2.0}),
    "cost_recovery": ("underweight", {"kind": "cost_recovery", "remaining_weight_pct": None}),
    "full": ("sell", {"kind": "full", "remaining_weight_pct": None}),
}
# (base plan, where the word lives, what the PM writes there, the word alpha-pulse must read):
# each word sits in a plan where it is natural, so only the word itself is under test.
VOCABULARY_CASES = (
    *(("minimal", ("rating",), word, word) for word in RATINGS),
    ("sell", ("tranches", 0, "trigger"), "immediate", "immediate"),
    ("sell", ("tranches", 1, "trigger"), "conditional", "conditional"),
    *(("sell", ("tranches", 1, "triggers", 0), trigger, kind)
      for kind, trigger in _TRIGGER_OF_KIND.items()),
    *((base, ("exit_target",), target, kind)
      for kind, (base, target) in _EXIT_TARGET_OF_KIND.items()),
    *(("sell", ("revision",), {"kind": kind, "note": "공시로 차환 금리가 확정됐다."}, kind)
      for kind in REVISION_KINDS),
)


def _without_prose(plan: dict[str, Any]) -> dict[str, Any]:
    return {k: copy.deepcopy(v) for k, v in plan.items() if k not in PROSE_FIELDS}


# name -> (PM raw tool args, what alpha-pulse must read back from the line)
ROUND_TRIP_PLANS: dict[str, tuple[Callable[[], dict], Callable[[], dict]]] = {
    # The Sell plan as a model wrote it before PlanRevision existed (the fixture has no
    # 'revision' key): revision now appears as an explicit null and nothing else changes.
    "sell_without_revision": (
        lambda: {**scenarios.sell_plan(),
                 "executive_summary": scenarios.REIT_SELL_EXEC_SUMMARY,
                 "investment_thesis": scenarios.REIT_SELL_THESIS},
        lambda: {**scenarios.sell_plan(), "revision": None},
    ),
    "sell_with_revision": (
        scenarios.reit_sell_pm,
        lambda: {**scenarios.sell_plan(), "revision": scenarios.REIT_REVISION},
    ),
    "samsung_buy": (BASE_PLANS["buy"], lambda: _without_prose(scenarios.SAMSUNG_BUY_PM)),
    "underweight_trailing_weight_exit": (BASE_PLANS["underweight"],
                                         lambda: _without_prose(UNDERWEIGHT_PM)),
}


@dataclass(frozen=True)
class NumberSlot:
    """A plan number alpha-pulse reads, and where the PM writes it."""

    label: str  # Model.field as the schema names it at a2981a7
    base: str  # key of BASE_PLANS
    path: tuple[str | int, ...]  # inside the PM's raw tool args (== inside the JSON line)
    plain: str  # a plain numeric string the model may send ...
    plain_value: float  # ... and the number alpha-pulse must read for it
    required: bool = False


NUMBER_SLOTS = (
    NumberSlot("PortfolioDecision.price_target", "sell", ("price_target",), "12050", 12050.0),
    NumberSlot("PortfolioDecision.stop_loss", "sell", ("stop_loss",), "12050", 12050.0),
    NumberSlot("PortfolioDecision.total_weight_pct", "buy", ("total_weight_pct",), "12.5", 12.5),
    NumberSlot("Tranche.pct", "sell", ("tranches", 0, "pct"), "12.5", 12.5, required=True),
    NumberSlot("Tranche.price_low", "sell", ("tranches", 0, "price_low"), "12050", 12050.0),
    NumberSlot("Tranche.price_high", "sell", ("tranches", 0, "price_high"), "12050", 12050.0),
    NumberSlot("TrancheTrigger.price", "sell", ("tranches", 1, "triggers", 0, "price"),
               "12050", 12050.0),
    NumberSlot("TrancheTrigger.trail_pct", "underweight",
               ("tranches", 1, "triggers", 0, "trail_pct"), "12.5", 12.5),
    NumberSlot("TrancheTrigger.reference_price", "sell",
               ("tranches", 1, "triggers", 0, "reference_price"), "12050", 12050.0),
    NumberSlot("ExitTarget.remaining_weight_pct", "underweight",
               ("exit_target", "remaining_weight_pct"), "12.5", 12.5),
    NumberSlot("KillSwitch.price", "sell", ("kill_switch", "price"), "12050", 12050.0),
)

# Numbers a model writes that cannot be read as ONE plain number. Upstream v0.5.1's
# _coerce_optional_float (1c44dd1/f8042ef) would turn them into null ('12,050원', '₩12,050',
# '3%', '150-160', 'around 12000', '1.2만') or strip the separator into a wrong number
# ('1,5' -> 15.0, '150,25' -> 15025.0, '11.500,00' -> 11.5, '1.234,50' -> 1.2345).
UNREADABLE_NUMBERS = ("12,050원", "₩12,050", "3%", "1,5", "150,25", "11.500,00", "1.234,50",
                      "150-160", "around 12000", "1.2만")
# How a model says "no value" (#1058): these alone may become null.
PLACEHOLDERS = ("N/A", "none", "-", "TBD", "", None)


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------

def _main_py_dump(plan_obj: pydantic.BaseModel) -> str:
    """Exactly the call main.py makes for the TRADE_PLAN_JSON line."""
    return plan_obj.model_dump_json(exclude={"executive_summary", "investment_thesis"})


def _validate_premise(plan_model: type[pydantic.BaseModel], raw: dict[str, Any],
                      what: str) -> pydantic.BaseModel:
    """Validate a plan the test relies on as a readable baseline; fail with context if it no
    longer validates (that alone means the PM would print no plan for it)."""
    try:
        return plan_model.model_validate(raw)
    except pydantic.ValidationError as exc:
        pytest.fail(f"premise: {what} no longer validates, so the PM would fall back to free text "
                    f"and print no TRADE_PLAN_JSON line for it:\n{exc}")


def _alpha_pulse_reads(line_json: str) -> dict[str, Any] | None:
    """What a reader of the documented line (``contract_reader``) makes of the printed line."""
    return contract_reader.read_trade_plan(PLAN_PREFIX + line_json)


def _with(data: dict[str, Any], path: tuple[str | int, ...], value: Any) -> dict[str, Any]:
    out = copy.deepcopy(data)
    target: Any = out
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = copy.deepcopy(value)
    return out


def _dig(data: Any, path: tuple[str | int, ...]) -> Any:
    for key in path:
        try:
            data = data[key]
        except (KeyError, IndexError, TypeError):
            return ABSENT
    return data


def _replace(obj: pydantic.BaseModel, path: tuple[str | int, ...], value: Any) -> Any:
    """Copy of ``obj`` with ``value`` at ``path``, bypassing validation (model_copy(update=))."""
    head, rest = path[0], path[1:]
    if not rest:
        return obj.model_copy(update={head: value})
    child = getattr(obj, str(head))
    if isinstance(rest[0], int):
        items = list(child)
        index = rest[0]
        items[index] = _replace(items[index], rest[1:], value) if rest[1:] else value
        return obj.model_copy(update={head: items})
    return obj.model_copy(update={head: _replace(child, rest, value)})


def _strip_annotated(tp: Any) -> Any:
    while typing.get_origin(tp) is typing.Annotated:
        tp = typing.get_args(tp)[0]
    return tp


def _describe(annotation: Any) -> tuple[str, bool, Any]:
    """(JSON kind, nullable, detail) of a field annotation.

    detail: the vocabulary (enum / Literal), the nested model class (object) or the item
    annotation (array). An Enum and a Literal with the same words dump identically, so both are
    "enum" here.
    """
    tp = _strip_annotated(annotation)
    nullable = False
    if typing.get_origin(tp) in (typing.Union, types.UnionType):
        members = typing.get_args(tp)
        concrete = [m for m in members if m is not type(None)]
        nullable = len(concrete) < len(members)
        if len(concrete) != 1:
            return f"union{concrete}", nullable, None
        tp = _strip_annotated(concrete[0])
    origin = typing.get_origin(tp)
    if origin is typing.Literal:
        return "enum", nullable, frozenset(typing.get_args(tp))
    if origin in (list, tuple, set, frozenset):
        args = typing.get_args(tp)
        return "array", nullable, (args[0] if args else None)
    if isinstance(tp, type):
        if issubclass(tp, enum.Enum):
            return "enum", nullable, frozenset(member.value for member in tp)
        if issubclass(tp, pydantic.BaseModel):
            return "object", nullable, tp
        for py_type, json_kind in ((bool, "boolean"), (int, "integer"), (float, "number"),
                                   (str, "string")):
            if tp is py_type:
                return json_kind, nullable, None
    return f"unrecognised {tp!r}", nullable, None


def _contract_problems(model: type[pydantic.BaseModel], contract: Mapping[str, Slot], where: str,
                       allowed_required: frozenset[str] = frozenset()) -> list[str]:
    problems: list[str] = []
    fields = model.model_fields
    for name, slot in contract.items():
        at = f"{where}.{name}"
        info = fields.get(name)
        if info is None:
            problems.append(f"{at}: gone from {model.__name__} (alpha-pulse reads this key)")
            continue
        kind, nullable, detail = _describe(info.annotation)
        if kind != slot.kind:
            problems.append(f"{at}: JSON type is {kind}, alpha-pulse reads {slot.kind}")
            continue
        if info.is_required() != slot.required:
            problems.append(f"{at}: required={info.is_required()}, contract required={slot.required}")
        if slot.nullable is not None and nullable != slot.nullable:
            problems.append(f"{at}: nullable={nullable}, contract nullable={slot.nullable}")
        if kind == "enum" and detail != frozenset(slot.values):
            problems.append(f"{at}: vocabulary {sorted(map(str, detail))} != alpha-pulse's "
                            f"{sorted(slot.values)}")
        if slot.fields is None:
            continue
        nested, nested_at = detail, at
        if kind == "array":
            item_kind, item_nullable, nested = _describe(detail)
            nested_at = f"{at}[]"
            if item_kind != "object" or item_nullable:
                problems.append(f"{nested_at}: items are {item_kind} (nullable={item_nullable}), "
                                "alpha-pulse reads objects")
                continue
        problems.extend(_contract_problems(nested, slot.fields, nested_at))
    for name, info in fields.items():
        if info.is_required() and name not in contract and name not in allowed_required:
            problems.append(f"{where}.{name}: new REQUIRED field -- every plan the model sends "
                            "without it fails validation and loses its TRADE_PLAN_JSON line")
    return problems


@pytest.fixture(scope="module")
def plan_model() -> type[pydantic.BaseModel]:
    """The Portfolio Manager's structured-output schema, imported from THIS checkout."""
    model = get_symbol("PortfolioDecision", *SCHEMA_MODULES)
    assert isinstance(model, type) and issubclass(model, pydantic.BaseModel), model
    source = Path(inspect.getsourcefile(model) or "").resolve()
    assert source.is_relative_to(REPO_ROOT.resolve()), (
        f"PortfolioDecision came from {source}, not from this checkout {REPO_ROOT} (an install "
        "of another checkout shadows it); every verdict below would be about the wrong code")
    return model


# ---------------------------------------------------------------------------------------------
# 1. shape: names, JSON types, required-ness, vocabularies
# ---------------------------------------------------------------------------------------------

def test_plan_fields_alpha_pulse_reads_keep_their_types_requiredness_and_vocabularies(plan_model):
    """Breaks if: a key alpha-pulse reads is renamed or dropped; changes JSON type (``seq`` as a
    float prints 1.0, which alpha-pulse's int check rejects; a price as a string/Decimal); turns
    required (every plan that omits it falls back to free text: no plan) or optional/nullable where
    alpha-pulse needs a value (a null rating/seq/pct/trigger/kind); a closed vocabulary gains or
    loses a word (a trigger or exit kind alpha-pulse cannot act on, a sixth rating); or a new
    REQUIRED field appears. New optional fields alpha-pulse does not read are allowed."""
    # Premise: the ratings of the plan are the tradeable words of the decision line.
    assert {w.lower() for w in RATINGS} == contract_reader.TRADEABLE_WORDS

    problems = _contract_problems(plan_model, PLAN_CONTRACT, "TRADE_PLAN_JSON",
                                  allowed_required=PROSE_FIELDS)
    assert not problems, "TRADE_PLAN_JSON schema drifted from what alpha-pulse reads:\n" + \
        "\n".join(problems)


def test_every_vocabulary_word_reaches_the_line_verbatim(plan_model):
    """Breaks if: the dump writes a vocabulary word differently from the word the model chose (an
    enum repr such as 'PortfolioRating.SELL', an enum NAME, a lower-cased or renamed word) or the
    schema stops accepting a word alpha-pulse acts on. alpha-pulse compares these strings exactly:
    the buy-side and sell-side ratings, 'immediate'/'conditional', the exit-target kinds and the
    revision kinds."""
    covered = {word for *_, word in VOCABULARY_CASES}
    assert covered == {*RATINGS, *TRANCHE_TRIGGERS, *TRIGGER_KINDS, *EXIT_TARGET_KINDS,
                       *REVISION_KINDS}, covered  # premise: every word is exercised
    wrong: list[str] = []
    for base, path, value, word in VOCABULARY_CASES:
        word_path = path if isinstance(value, str) else (*path, "kind")
        try:
            obj = plan_model.model_validate(_with(BASE_PLANS[base](), path, value))
        except pydantic.ValidationError as exc:
            wrong.append(f"{word_path} {word!r}: rejected ({exc.errors()[0]['msg']})")
            continue
        read = _alpha_pulse_reads(_main_py_dump(obj))
        got = ABSENT if read is None else _dig(read, word_path)
        if got != word:
            wrong.append(f"{word_path} {word!r}: alpha-pulse reads {got!r}")
    assert not wrong, "\n".join(wrong)


# ---------------------------------------------------------------------------------------------
# 2. reviewed plans round-trip through validation + main.py's dump call
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(ROUND_TRIP_PLANS))
def test_reviewed_plans_round_trip_through_the_main_py_dump_value_for_value(plan_model, name):
    """Breaks if: a reviewed plan no longer validates (a new required field, a narrowed type or
    vocabulary: the PM would fall back to free text and alpha-pulse would get no plan), or
    validation + main.py's dump call changes anything alpha-pulse reads: a coercer rewriting a
    number, a serializer renaming/dropping keys or omitting null-valued keys (alpha-pulse rejects a
    kill_switch without a 'price' key), int/float drift, the revision key missing instead of null,
    or the line no longer being one line of strict JSON (prose with line breaks included)."""
    make_raw, make_expected = ROUND_TRIP_PLANS[name]
    obj = _validate_premise(plan_model, make_raw(), f"the reviewed plan {name!r}")
    line = _main_py_dump(obj)
    assert "\n" not in line and "\r" not in line, line[:200]
    read = _alpha_pulse_reads(line)
    assert read is not None, f"the line does not read as one JSON object: {line[:300]}"
    expected = make_expected()
    assert canonical_json(project_to_expected(read, expected)) == canonical_json(expected)


def test_plan_shape_of_before_the_2026_08_trigger_change_is_rejected_whole(plan_model):
    """Breaks if: the schema starts accepting the plan shape of before the 2026-08 trigger change
    again (docs/INTEGRATION.md "Breaking, 2026-08": a tranche ``trigger`` of 'price' / 'event',
    the condition in prose only, no ``triggers[]``) or reinterprets it -- e.g. a before-validator
    mapping 'price'/'event' onto 'conditional' with no condition objects, or dropping the old
    tranches and keeping the rest. A model that still writes the old shape must lose the WHOLE
    plan (free-text fallback: no TRADE_PLAN_JSON line), never have it printed with a trigger word
    outside the two alpha-pulse acts on or with tranches missing (pct would no longer sum to
    100). Input: a synthetic plan in the old shape (fixtures/synthetic_legacy_underweight_plan
    .json) plus the two required prose fields; control: the same position restated in today's
    shape (UNDERWEIGHT_PM) validates."""
    legacy = scenarios.fixture_json("synthetic_legacy_underweight_plan.json")
    # Premise: the fixture really is the old shape.
    assert [t["trigger"] for t in legacy["tranches"]] == ["immediate", "price", "event"]
    assert all("triggers" not in t for t in legacy["tranches"]), legacy["tranches"]
    _validate_premise(plan_model, UNDERWEIGHT_PM, "the same position in today's shape")

    raw = {**legacy, "executive_summary": UNDERWEIGHT_PM["executive_summary"],
           "investment_thesis": UNDERWEIGHT_PM["investment_thesis"]}
    try:
        obj = plan_model.model_validate(raw)
    except pydantic.ValidationError as exc:
        rejected = {tuple(err["loc"]) for err in exc.errors()}
        # Rejected for the old trigger words themselves, not for something incidental.
        assert {("tranches", 1, "trigger"), ("tranches", 2, "trigger")} <= rejected, rejected
        return
    pytest.fail("the old-shape plan validated; the line would read: "
                f"{_alpha_pulse_reads(_main_py_dump(obj))}")


# ---------------------------------------------------------------------------------------------
# 3. non-finite numbers
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("slot", NUMBER_SLOTS, ids=lambda s: s.label)
def test_non_finite_plan_numbers_never_reach_the_line_as_nan_or_infinity(plan_model, slot):
    """Breaks if: the plan schema's serializer writes a non-finite number as a NaN/Infinity token
    (ser_json_inf_nan='constants', a custom serializer) -- they are not JSON, and a strict reader
    of the line rejects them, so a plan with one such number silently turns 'unparsed' -- or as
    a string ('strings' mode) that alpha-pulse cannot use as a number; or a coercer turns
    'NaN'/'Infinity' from the model into a finite number (a made-up level instead of none). Today
    a non-finite value is written as null; failing the whole plan instead would also satisfy
    alpha-pulse. (This is the schema's serializer; main.py's own dump call is locked end to end,
    for a finite plan, by test_main_py_prints_the_pm_plan_exactly_as_the_schema_dumps_it.)"""
    base_obj = _validate_premise(plan_model, BASE_PLANS[slot.base](), f"the {slot.base} plan")
    wrong: dict[str, Any] = {}
    for value in (math.nan, math.inf, -math.inf):
        line = _main_py_dump(_replace(base_obj, slot.path, value))
        read = _alpha_pulse_reads(line)
        if read is None:
            wrong[f"serialized {value!r}"] = f"alpha-pulse cannot parse the line: {line[:160]}"
        elif _dig(read, slot.path) is not None:
            wrong[f"serialized {value!r}"] = _dig(read, slot.path)
    for raw in ("NaN", "nan", "Infinity", "-inf", math.nan, math.inf):
        try:
            obj = plan_model.model_validate(_with(BASE_PLANS[slot.base](), slot.path, raw))
        except pydantic.ValidationError:
            continue  # fail-closed is fine
        read = _alpha_pulse_reads(_main_py_dump(obj))
        got = "<line unparseable>" if read is None else _dig(read, slot.path)
        if got is not None:
            wrong[f"model sent {raw!r}"] = got
    assert not wrong, f"{slot.label}: {wrong}"


# ---------------------------------------------------------------------------------------------
# 4. fail-closed numbers: all or nothing
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("slot", NUMBER_SLOTS, ids=lambda s: s.label)
def test_unreadable_plan_number_fails_the_whole_plan(plan_model, slot):
    """Breaks if: an unreadable plan number validates instead of failing the whole plan --
    upstream v0.5.1's _coerce_optional_float (1c44dd1/f8042ef, auto-merged outside any conflict
    marker onto every fork validator) nulls '12,050원'/'3%'/'150-160'/'around 12000' and strips
    separators ('1,5' -> 15.0, ten times the 1.5 meant; '11.500,00' -> 11.5, a stop nobody
    reaches); a KRW/percent-parsing coercer reads '12,050원' as 12050.0 or '3%' as 3.0; a
    salvage step drops the bad field or tranche and keeps the rest. alpha-pulse would then store
    a plan with a null or wrong number as 'ok' and draft orders from it, instead of 'unparsed'
    with zero drafts."""
    base = BASE_PLANS[slot.base]()
    # Control: the same plan with a plain number in that place validates, so every failure below
    # is caused by the unreadable value alone.
    _validate_premise(plan_model, _with(base, slot.path, slot.plain),
                      f"the {slot.base} plan with {slot.label}={slot.plain!r}")
    leaked: dict[str, Any] = {}
    for raw in UNREADABLE_NUMBERS:
        try:
            obj = plan_model.model_validate(_with(base, slot.path, raw))
        except pydantic.ValidationError:
            continue
        read = _alpha_pulse_reads(_main_py_dump(obj))
        leaked[raw] = "<line unparseable>" if read is None else _dig(read, slot.path)
    assert not leaked, (
        f"{slot.label}: unreadable values validated instead of failing the whole plan "
        f"(model wrote -> alpha-pulse would read): {leaked}")


@pytest.mark.parametrize("slot", NUMBER_SLOTS, ids=lambda s: s.label)
def test_placeholders_mean_not_provided_and_plain_numeric_strings_parse(plan_model, slot):
    """Breaks if: a placeholder the model writes for "no value" ('N/A', 'none', '-', 'TBD', '',
    null) stops meaning null for an optional number (a stricter schema turns each such plan into a
    free-text fallback: no plan at all), or a plain numeric string stops parsing to its number
    (strict mode, a coercer rewriting it). Tranche.pct is required -- the tranche's share of the
    move -- so a placeholder there must fail the plan: alpha-pulse reads a null pct as 0% and
    would silently re-partition the tranches."""
    base = BASE_PLANS[slot.base]()
    obj = _validate_premise(plan_model, _with(base, slot.path, slot.plain),
                            f"the {slot.base} plan with {slot.label}={slot.plain!r}")
    read = _alpha_pulse_reads(_main_py_dump(obj))
    got = ABSENT if read is None else _dig(read, slot.path)
    assert isinstance(got, float) and got == slot.plain_value, (slot.label, slot.plain, got)

    wrong: dict[str, str] = {}
    for raw in PLACEHOLDERS:
        try:
            obj = plan_model.model_validate(_with(base, slot.path, raw))
        except pydantic.ValidationError:
            if not slot.required:
                wrong[repr(raw)] = "ValidationError: the whole plan would be lost"
            continue
        read = _alpha_pulse_reads(_main_py_dump(obj))
        got = ABSENT if read is None else _dig(read, slot.path)
        if slot.required:
            wrong[repr(raw)] = f"validated as {got!r}: a required share left unstated"
        elif got is not None:
            wrong[repr(raw)] = f"alpha-pulse reads {got!r}, not null"
    assert not wrong, f"{slot.label}: {wrong}"


# ---------------------------------------------------------------------------------------------
# 5. end to end: the real main.py, as alpha-pulse runs it
# ---------------------------------------------------------------------------------------------

def test_main_py_prints_the_pm_plan_exactly_as_the_schema_dumps_it(ap_run):
    """Breaks if: main.py changes how it prints the plan -- exclude_none (null keys vanish;
    alpha-pulse rejects a kill_switch without a 'price' key), indent (a multi-line line a one-line
    reader never matches: no plan), aliases/renamed keys, a second TRADE_PLAN_JSON line, or a dump
    of something other than the Portfolio Manager's typed decision. The expected line is the
    synthetic Sell plan (fixture) plus the revision the PM sent in s1."""
    res = ap_run(S1)
    res.assert_ok()
    assert len(res.trade_plan_lines) == 1, (
        f"main.py printed {len(res.trade_plan_lines)} lines of the TRADE_PLAN_JSON form (a "
        f"consumer needs exactly one):\n{res.describe()}")
    assert res.plan is not None, res.describe()
    expected = {**scenarios.sell_plan(), "revision": scenarios.REIT_REVISION}
    assert canonical_json(project_to_expected(res.plan, expected)) == canonical_json(expected)


E2E_UNREADABLE = {
    # The stop written the way a Korean prompt invites it: currency suffix, thousands separator.
    "stop_loss_krw": (("stop_loss",), "12,050원"),
    # The same inside a conditional tranche's take-profit trigger (the Sell plan's 11,060).
    "take_profit_trigger_price_krw": (("tranches", 1, "triggers", 0, "price"), "11,060원"),
}


@pytest.mark.parametrize("case", sorted(E2E_UNREADABLE))
def test_unreadable_pm_plan_number_prints_no_trade_plan_line_and_still_exits_0(ap_run, case):
    """Breaks if: an unreadable number in the Portfolio Manager's plan no longer discards the plan
    end to end -- a lenient coercer (upstream v0.5.1 turns '12,050원' into null), a salvage step in
    the PM node or schema, or main.py printing a partial plan. alpha-pulse would then store
    plan_status 'ok' and draft orders with no stop (or a take-profit without a price) instead of
    'unparsed' with zero drafts. The run itself must still succeed: rc 0 with the report saved
    (rc != 0 would throw the whole analysis away)."""
    path, raw = E2E_UNREADABLE[case]
    control = ap_run(S1)  # the same holding and plan with readable numbers prints a plan
    assert control.plan is not None, (
        "premise: s1 (the same plan with readable numbers) printed no plan alpha-pulse can read, "
        f"so this case would prove nothing:\n{control.describe()}")

    pm = _with(scenarios.reit_sell_pm(), path, raw)
    spec = scenarios.derive(scenarios.get(S1),
                            {"roles": {"portfolio_manager": {"structured": pm}}},
                            name=f"tp_unreadable_{case}", memory_log_seed=None)
    res = ap_run(spec)
    res.assert_ok()

    # Premise: the PM's structured reply really carried the unreadable number.
    structured = res.calls_for("portfolio_manager", "structured")
    assert structured, f"the PM made no structured call; roles: {sorted(res.roles_called())}"
    sent = structured[0]["reply"]["tool_calls"][0]["args"]
    assert _dig(sent, path) == raw, _dig(sent, path)

    # The contract: no plan line at all -- not a plan with that number nulled or rewritten.
    assert not [ln for ln in res.stdout_lines if ln.lstrip().startswith("TRADE_PLAN_JSON")], \
        res.trade_plan_lines
    assert res.plan is None

    # ...while the analysis itself stays usable: a completed run (rc 0, the report line read)
    # that alpha-pulse stores with "no plan".
    assert (res.rc, res.report_path is not None) == (0, True), res.describe()
    assert REPORT_SAVED_LINE_RE.match(res.stdout_lines[-1]), res.stdout_lines[-3:]
    assert res.complete_report, "complete_report.md missing"
    assert (res.decision or "").lower() in DECISION_WORDS, res.decision
