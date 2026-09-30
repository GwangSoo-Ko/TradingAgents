"""Clean-room reader of the lines main.py prints for a program that shells out to it.

Written from the fork's own public description of main.py's output -- not from any
consumer's code:

* ``docs/INTEGRATION.md`` §1b "Running main.py and consuming its output": main.py prints
  the decision, then -- only when the Portfolio Manager's structured call produced a plan
  -- one ``TRADE_PLAN_JSON: {...}`` line, then ``Report saved: <path>``; a consumer reads
  the report path from that line (it does not glob for it), and when the plan line is
  absent there is no plan -- it must not synthesise one from the prose;
* main.py's comments at the ``TRADE_PLAN_JSON`` print: the line is
  ``"TRADE_PLAN_JSON: " + PortfolioDecision.model_dump_json(exclude=...)`` on ONE line, and
  a consumer that cannot read it stores "no plan" (``plan_status='unparsed'``) -- nothing
  raises on either side;
* the fork's REVIEW sentinel (``tradingagents/agents/utils/rating.py``: ``RATING_REVIEW``,
  ``is_review``): the signal main.py prints when the decision names no rating. It asks for
  a human look or a re-run; it is not a tradeable rating.

What a line is: main.py writes each of these lines with ``print()``, so a line is the text
up to a ``"\\n"`` (a ``"\\r"`` just before it belongs to the line ending). Nothing else ends
a line. JSON leaves U+2028 / U+2029 / U+0085 inside strings unescaped and pydantic's dump
prints them raw, so a plan whose prose holds one is still ONE line; a lone ``"\\r"`` or a
form feed is an ordinary character. ``str.splitlines()`` would cut such a line in two.

main.py builds its graph with ``debug=True``, so agent prose streams to stdout BEFORE these
closing lines, and prose can hold a bare rating word or even a line that looks like
``Report saved: ...``. The grammar therefore anchors on the last occurrence:

* report path -- the text after ``Report saved:`` on the LAST line that starts with it and
  names a path, surrounding whitespace stripped (``Report saved:`` followed by nothing but
  whitespace names no report and is not such a line);
* decision -- walking up from that line (from the end of the output when there is no such
  line), the first line whose stripped, lower-cased text is one of ``buy``,
  ``overweight``, ``hold``, ``underweight``, ``sell`` or ``review``; returned stripped,
  case kept;
* plan -- the LAST line of the form ``TRADE_PLAN_JSON: {...}`` (the prefix at the very
  start of the line, then a ``{...}`` and nothing else but whitespace), read as one strict
  JSON object. A ``TRADE_PLAN_JSON:`` line without that form (nothing after the prefix, an
  array, text after the closing brace) is not a plan line. ``NaN`` / ``Infinity`` /
  ``-Infinity`` are not JSON; a plan line whose text does not parse, or parses to anything
  but an object, means no plan (``None``) -- never a partial plan and never an earlier line
  instead.
"""

from __future__ import annotations

import json
from typing import Any

REPORT_SAVED_PREFIX = "Report saved:"
TRADE_PLAN_PREFIX = "TRADE_PLAN_JSON:"

# The five ratings of the fork's scale, in order, as the decision line spells them
# (compared lower-cased), and the REVIEW sentinel, which is a decision but not a rating.
RATING_WORDS = ("buy", "overweight", "hold", "underweight", "sell")
REVIEW = "review"
TRADEABLE_WORDS = frozenset(RATING_WORDS)
DECISION_WORDS = frozenset((*RATING_WORDS, REVIEW))


def lines(stdout: str | None) -> list[str]:
    """``stdout`` cut into the lines ``print()`` wrote (see the module doc).

    Split at every ``"\\n"`` only; a ``"\\r"`` right before it is dropped; output that ends
    with a newline has no empty last line (``"a\\n"`` is one line, ``""`` none).
    """
    if not stdout:
        return []
    pieces = stdout.split("\n")
    if pieces[-1] == "":
        pieces.pop()
    return [piece[:-1] if piece.endswith("\r") else piece for piece in pieces]


def report_path_of(line: str) -> str | None:
    """The path a ``Report saved: <path>`` line names; None for any other line, including
    a ``Report saved:`` line that names no path."""
    if not line.startswith(REPORT_SAVED_PREFIX):
        return None
    path = line[len(REPORT_SAVED_PREFIX):].strip()
    return path if path else None


def plan_text_of(line: str) -> str | None:
    """The ``{...}`` text of a ``TRADE_PLAN_JSON: {...}`` line; None for any other line."""
    if not line.startswith(TRADE_PLAN_PREFIX):
        return None
    text = line[len(TRADE_PLAN_PREFIX):].strip()
    return text if text[:1] == "{" and text[-1:] == "}" else None


def is_plan_line(line: str) -> bool:
    return plan_text_of(line) is not None


def _last_report_line(rows: list[str]) -> tuple[int, str] | None:
    """(index, path) of the last line that names a report; None when no line does."""
    for index in range(len(rows) - 1, -1, -1):
        path = report_path_of(rows[index])
        if path is not None:
            return index, path
    return None


def read_stdout(stdout: str | None) -> tuple[str | None, str | None]:
    """(decision, report path) as the documented grammar defines them (see module doc)."""
    rows = lines(stdout)
    found = _last_report_line(rows)
    above, report_path = (rows[:found[0]], found[1]) if found else (rows, None)
    for row in reversed(above):
        word = row.strip()
        if word.lower() in DECISION_WORDS:
            return word, report_path
    return None, report_path


def _not_json(literal: str) -> Any:
    raise ValueError(f"{literal} is not a JSON number")


# Strict JSON: the three non-finite literals Python's json module would accept are refused.
_STRICT_JSON = json.JSONDecoder(parse_constant=_not_json)


def read_trade_plan(stdout: str | None) -> dict[str, Any] | None:
    """The plan of the LAST ``TRADE_PLAN_JSON: {...}`` line, or None (see module doc)."""
    for row in reversed(lines(stdout)):
        text = plan_text_of(row)
        if text is not None:
            break
    else:
        return None
    try:
        value = _STRICT_JSON.decode(text)
    except ValueError:  # json.JSONDecodeError is a ValueError too
        return None
    return value if isinstance(value, dict) else None


def is_review(signal: str | None) -> bool:
    """True when a decision line carries the REVIEW sentinel ("look at it", not a trade)."""
    if signal is None:
        return False
    return signal.strip().lower() == REVIEW
