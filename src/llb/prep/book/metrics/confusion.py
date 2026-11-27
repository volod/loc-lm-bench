"""Dangerous substitutions, one count per confusion class.

A class fires on a substitution inside its set, or on a deletion or insertion of a sign, a
bracket, a unit symbol, or a Greek letter. A deleted digit is ordinary missing text: ``5`` becomes
``S`` counts as ``5/S``, and ``5`` disappearing does not. Equation numbers and sub/superscripts
are counted from their own structure so a changed ``(3-4)`` is not also a handful of brackets.
"""

import re

from llb.prep.book.metrics.align import alignment
from llb.prep.book.metrics.constants import (
    CLASS_BRACKET,
    CLASS_DECIMAL,
    CLASS_EQUATION,
    CLASS_FIVE_S,
    CLASS_LATIN_GREEK,
    CLASS_MINUS,
    CLASS_ONE_LI,
    CLASS_SCRIPT,
    CLASS_UNIT,
    CLASS_ZERO_O,
    CONFUSION_CLASSES,
)

_EQUATION = re.compile(r"\(\d{1,4}(?:[-\u2212\u2010\u2011\u2013\u2014]\d{1,4})+\)")
_MINUS = frozenset("-\u2212\u2010\u2011\u2013\u2014")
_BRACKETS = frozenset("()[]{}<>\u27e8\u27e9\u3008\u3009")
_ONE = frozenset({"1", "l", "I", "|"})
_FIVE = frozenset({"5", "S", "s"})
_DIGITS = frozenset("0123456789")
_UNIT_PAIRS = (
    frozenset({"\u00b0", "o"}),
    frozenset({"\u00b0", "0"}),
    frozenset({"\u00b5", "u"}),
    frozenset({"\u00b5", "\u03bc"}),
    frozenset({"\u03a9", "O"}),
    frozenset({"\u212b", "A"}),
    frozenset({"\u2030", "%"}),
)
_UNIT_SYMBOLS = frozenset("\u00b0\u00b5\u03a9\u212b\u2030")
_GREEK_PAIRS = tuple(
    frozenset(pair)
    for pair in (
        "a\u03b1",
        "A\u0391",
        "B\u0392",
        "d\u03b4",
        "D\u0394",
        "e\u03b5",
        "E\u0395",
        "g\u03b3",
        "h\u03b7",
        "H\u0397",
        "i\u03b9",
        "k\u03ba",
        "l\u03bb",
        "n\u03bd",
        "N\u039d",
        "o\u03bf",
        "O\u039f",
        "p\u03c1",
        "r\u03c1",
        "t\u03c4",
        "u\u03bc",
        "v\u03bd",
        "w\u03c9",
        "x\u03c7",
        "y\u03b3",
        "z\u03b6",
    )
)
_GREEK = frozenset(char for pair in _GREEK_PAIRS for char in pair if ord(char) > 127)
_SUB_CHARS = frozenset(
    "\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089"
    "\u208a\u208b\u208c\u208d\u208e\u2090\u2091\u2092\u2093\u2094"
    "\u2095\u2096\u2097\u2098\u2099\u209a\u209b\u209c"
)
_SCRIPT_FROM = (
    "\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089"
    "\u208a\u208b\u208c\u208d\u208e\u2090\u2091\u2092\u2093\u2094"
    "\u2095\u2096\u2097\u2098\u2099\u209a\u209b\u209c"
    "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079"
    "\u207a\u207b\u207c\u207d\u207e\u207f\u2071"
)
_SCRIPT_TO = "0123456789+-=()aeox\u0259hklmnpst0123456789+-=()ni"
_SCRIPT_BASE = str.maketrans(_SCRIPT_FROM, _SCRIPT_TO)
_SCRIPT = frozenset(
    "\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089"
    "\u208a\u208b\u208c\u208d\u208e\u2090\u2091\u2092\u2093\u2094"
    "\u2095\u2096\u2097\u2098\u2099\u209a\u209b\u209c"
    "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079"
    "\u207a\u207b\u207c\u207d\u207e\u207f\u2071"
)


def empty_counts() -> dict[str, int]:
    """Every class, present and zero, so a report cannot drop a class by omitting it."""
    return {name: 0 for name in CONFUSION_CLASSES}


def confusion_counts(reference: str, hypothesis: str, script_events: int) -> dict[str, int]:
    """Per-class counts for one page. ``script_events`` comes from the formula structure."""
    ref_masked, hyp_masked, equation_events = _mask_equations(reference, hypothesis)
    ref_text, hyp_text = _strip_scripts(ref_masked), _strip_scripts(hyp_masked)
    counts = empty_counts()
    for op, left, right in alignment(ref_text, hyp_text):
        kind = _operation_class(op, left, right, ref_text, hyp_text)
        if kind is not None:
            counts[kind] += 1
    counts[CLASS_EQUATION] += equation_events
    counts[CLASS_SCRIPT] += script_events
    return counts


def equation_numbers(text: str) -> list[str]:
    """Equation numbers in reading order, parentheses included."""
    return [match.group() for match in _EQUATION.finditer(text)]


def unicode_script_tokens(text: str) -> list[tuple[str, str, str]]:
    """``(base, subscript, superscript)`` runs printed with Unicode script characters."""
    tokens: list[tuple[str, str, str]] = []
    index = 0
    while index < len(text):
        if text[index] in _SCRIPT and index > 0 and text[index - 1] not in _SCRIPT:
            base = text[index - 1]
            sub: list[str] = []
            sup: list[str] = []
            while index < len(text) and text[index] in _SCRIPT:
                (sub if text[index] in _SUB_CHARS else sup).append(
                    text[index].translate(_SCRIPT_BASE)
                )
                index += 1
            tokens.append((base, "".join(sub), "".join(sup)))
            continue
        index += 1
    return tokens


def script_events(
    reference: list[tuple[str, str, str]], hypothesis: list[tuple[str, str, str]]
) -> int:
    """Changed, missing, and extra scripts. One event per script, not per character."""
    remaining = list(hypothesis)
    events = 0
    for token in reference:
        if token in remaining:
            remaining.remove(token)
            continue
        same_base = next(
            (index for index, item in enumerate(remaining) if item[0] == token[0]), None
        )
        if same_base is None:
            events += 1
            continue
        remaining.pop(same_base)
        events += 1
    return events + len(remaining)


def normalize_equation(number: str) -> str:
    """Digit key of an equation number. Dash lookalikes compare equal."""
    return re.sub(r"[^\d]", "-", number.strip())


def masked_pair(reference: str, hypothesis: str) -> tuple[str, str]:
    """The character strings the confusion walk sees, equation numbers and scripts aside."""
    ref_masked, hyp_masked, _events = _mask_equations(reference, hypothesis)
    return _strip_scripts(ref_masked), _strip_scripts(hyp_masked)


def step_class(op: str, left: int, right: int, reference: str, hypothesis: str) -> str | None:
    """The confusion class of one alignment step, or None when the step is not one."""
    return _operation_class(op, left, right, reference, hypothesis)


def is_script_character(char: str) -> bool:
    """True for a Unicode sub/superscript the script confusion class counts."""
    return len(char) == 1 and char in _SCRIPT


def _mask_equations(reference: str, hypothesis: str) -> tuple[str, str, int]:
    """Drop equation numbers whose digits differ, and count each such number once.

    A number whose digits match stays in the text, so a minus written as a hyphen still counts
    in the minus class.
    """
    ref_spans = list(_EQUATION.finditer(reference))
    hyp_spans = list(_EQUATION.finditer(hypothesis))
    drop_ref: list[tuple[int, int]] = []
    drop_hyp: list[tuple[int, int]] = []
    events = 0
    for index in range(max(len(ref_spans), len(hyp_spans))):
        ref_span = ref_spans[index] if index < len(ref_spans) else None
        hyp_span = hyp_spans[index] if index < len(hyp_spans) else None
        same = (
            ref_span is not None
            and hyp_span is not None
            and normalize_equation(ref_span.group()) == normalize_equation(hyp_span.group())
        )
        if same:
            continue
        events += 1
        if ref_span is not None:
            drop_ref.append(ref_span.span())
        if hyp_span is not None:
            drop_hyp.append(hyp_span.span())
    return _drop(reference, drop_ref), _drop(hypothesis, drop_hyp), events


def _drop(text: str, spans: list[tuple[int, int]]) -> str:
    kept = []
    cursor = 0
    for start, end in spans:
        kept.append(text[cursor:start])
        cursor = end
    kept.append(text[cursor:])
    return "".join(kept)


def _strip_scripts(text: str) -> str:
    return "".join(char for char in text if char not in _SCRIPT)


def _operation_class(op: str, left: int, right: int, reference: str, hypothesis: str) -> str | None:
    if op == "eq":
        return None
    if op == "sub":
        return _pair_class(reference[left], hypothesis[right])
    if op == "del":
        return _lone_class(reference[left], reference, left)
    return _lone_class(hypothesis[right], hypothesis, right)


def _pair_class(left: str, right: str) -> str | None:
    if left == right:
        return None
    pair = frozenset({left, right})
    if "0" in pair and pair <= frozenset("0Oo"):
        return CLASS_ZERO_O
    if pair <= _ONE and len(pair) == 2:
        return CLASS_ONE_LI
    if pair <= _FIVE and len(pair) == 2:
        return CLASS_FIVE_S
    if pair <= (_MINUS | {"+"}) and len(pair) == 2:
        return CLASS_MINUS
    if pair in _UNIT_PAIRS:
        return CLASS_UNIT
    if pair in _GREEK_PAIRS:
        return CLASS_LATIN_GREEK
    if pair <= _BRACKETS and len(pair) == 2:
        return CLASS_BRACKET
    if pair == frozenset({".", ","}):
        return CLASS_DECIMAL
    return None


def _lone_class(char: str, text: str, index: int) -> str | None:
    if char in _MINUS:
        return CLASS_MINUS
    if char in _BRACKETS:
        return CLASS_BRACKET
    if char in _UNIT_SYMBOLS:
        return CLASS_UNIT
    if char in _GREEK:
        return CLASS_LATIN_GREEK
    if char in ".," and _between_digits(text, index):
        return CLASS_DECIMAL
    return None


def _between_digits(text: str, index: int) -> bool:
    return (
        index > 0
        and index + 1 < len(text)
        and text[index - 1] in _DIGITS
        and text[index + 1] in _DIGITS
    )
