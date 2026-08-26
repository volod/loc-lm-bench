"""Two written forms of one VALUE: `2,9 млн осіб` and `2.9 мільйона осіб` are the same population.

The gate folds an answer's declared endpoints onto the corpus's own surfaces before it compares
them, and the only fold it had was the alias map the extraction ledger happens to record. That
leaves every `functional`, `inverse_functional`, and `max_cardinality` axiom broken wherever a
value has more than one written form -- which is exactly where a model paraphrases: numbers,
dates, durations. Nothing in the ledger has a reason to record `2,9 млн осіб` as an alias of
`2.9 мільйона осіб`, so the gate read the answer's restatement as a SECOND value of a functional
relation and refused a correct answer.

This module answers the one question that fixes: for the three closed-vocabulary types whose
members are VALUES rather than names -- `QUANTITY`, `DATE`, `DURATION` -- what value does a
surface denote? Two surfaces sharing a value key are one value. A surface this module cannot parse
has NO key and folds exactly as it did before, so the fix can never be worse than the alias map.

Three boundaries are deliberate:

  - **Only value types.** `Київ` is a name, not a value, and never gets a key. Identity between
    NAMES is what the corpus's alias map and the resolution overlay decide
    (`llb.graph.resolution.overlay`); a second notion of it here is what the scope boundary of
    this work forbids. `MONEY` is left out for the same reason a reviewer would want: a sum
    carries a currency this module would have to invent a table for.
  - **Refuse rather than guess.** A surface with two numbers in it, a duration with no time unit,
    or a date with a word the date grammar does not know parses to nothing. A wrong key would
    merge two values that differ, which loses a planted violation -- the one failure this gate may
    not have.
  - **Units come from a CLOSED table, not from the lemmatizer.** The pinned analyzer maps `рік`
    to `ріка` (the river), so `20 років` and `20 рік` would part company under it. The lemmatizer
    still handles the open tail, where it is right: `осіб` -> `особа`. That table family --
    magnitudes, units, cardinals, month names -- lives in `value_lexicon`, so extending the
    Ukrainian this reads never means reading the parser.
"""

import re
from collections.abc import Callable
from decimal import Decimal, InvalidOperation

from llb.eval.answer_validation.value_lexicon import (
    DATE_FILLER_STEMS,
    PERCENT,
    MONTHS,
    NUMERALS,
    TIME_UNITS,
    magnitude,
    stem_match,
    unit,
)
from llb.prep.ontology.extraction.entity_types import DATE, DURATION, QUANTITY

# One surface token -> its lemma. Injected so the module is testable without an analyzer; the
# default is the pinned Ukrainian lemmatizer the lexical index and the answer-span scorer use.
Lemmatizer = Callable[[str], str]

# The types this module can key. Everything else returns None and folds as before.
VALUE_TYPES: tuple[str, ...] = (QUANTITY, DATE, DURATION)

_APOSTROPHES = str.maketrans({variant: "'" for variant in "'‘’ʼ`"})
_SPACES = str.maketrans({variant: " " for variant in "    "})

# A literal number: grouped thousands first (`2 900 000`), then a plain decimal (`2,9` / `2.9`).
# Ukrainian writes the decimal separator as a comma and the thousands separator as a space, so
# both `,` and `.` read as decimal here -- `1.500` keys as 1.5, which is the Ukrainian reading.
_NUMBER_RE = re.compile(r"\d{1,3}(?: \d{3})+|\d+(?:[.,]\d+)?")
_WORD_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*")
_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_DOTTED_DATE_RE = re.compile(r"\b(\d{1,2})[./](\d{1,2})[./](\d{4})\b")

_YEAR_RANGE = range(1000, 3000)
_MAX_DAY = 31
_UNKNOWN = "?"


def value_key(surface: str, entity_type: str, lemmatize: Lemmatizer | None = None) -> str | None:
    """The value `surface` denotes when read as `entity_type`, or None when it does not parse.

    The key is type-scoped: `1990 рік` read as a `DATE` and read as a `QUANTITY` are different
    claims, and folding them together would be exactly the invented identity this gate refuses.
    """
    if entity_type == QUANTITY:
        return _quantity_key(surface, lemmatize or default_lemmatizer())
    if entity_type == DURATION:
        return _duration_key(surface)
    if entity_type == DATE:
        return _date_key(surface)
    return None


def default_lemmatizer() -> Lemmatizer:
    """The pinned Ukrainian lemmatizer, resolved lazily so importing this module stays cheap."""
    from llb.rag.vector_store.lexical import ukrainian_lemma

    return ukrainian_lemma


def _fold(surface: str) -> str:
    """Casefold, unify apostrophe variants, and turn every space variant into a plain space."""
    return " ".join(surface.translate(_APOSTROPHES).translate(_SPACES).casefold().split())


def _to_decimal(literal: str) -> Decimal | None:
    try:
        return Decimal(literal.replace(" ", "").replace(",", "."))
    except InvalidOperation:
        return None


def _split(text: str) -> tuple[list[Decimal], list[str]]:
    """The literal numbers in `text` and the word tokens left once they are taken out."""
    numbers: list[Decimal] = []

    def take(match: re.Match[str]) -> str:
        parsed = _to_decimal(match.group(0))
        if parsed is None:
            return match.group(0)
        numbers.append(parsed)
        return " "

    return numbers, _WORD_RE.findall(_NUMBER_RE.sub(take, text))


def _amount(value: Decimal) -> str:
    """The number as a stable string: `2.90` and `2.9` are one value, and so are `2` and `2.0`."""
    normalized = value.normalize()
    return f"{normalized:f}"


def _quantity_key(surface: str, lemmatize: Lemmatizer) -> str | None:
    """`<number> [magnitude] <unit words>` -- `2,9 млн осіб` and `2 900 000 осіб` key alike.

    Every word that is not a magnitude joins the unit, lemmatized when the closed table has no
    reading for it. Nothing is dropped: `близько 3 млн осіб` keeps `близько` and therefore does
    NOT fold onto `3 млн осіб`, which is the conservative reading of a hedged value.
    """
    folded = _fold(surface)
    numbers, words = _split(folded)
    if len(numbers) != 1:
        return None
    amount, units = numbers[0], []
    for word in words:
        factor = magnitude(word)
        if factor is not None:
            amount *= factor
            continue
        units.append(unit(word) or lemmatize(word))
    if PERCENT in folded:
        units.append(PERCENT)
    return f"{QUANTITY.lower()}:{_amount(amount)}:{'+'.join(sorted(set(units)))}"


def _duration_key(surface: str) -> str | None:
    """`<number or cardinal> <time unit>` -- `двадцять років` and `20 років` key alike.

    Exactly one time unit and nothing unaccounted for, because a duration the grammar only half
    reads is a value this module does not know.
    """
    numbers, words = _split(_fold(surface))
    if len(numbers) > 1:
        return None
    amount = numbers[0] if numbers else Decimal(0)
    spelled, units = 0, []
    for word in words:
        factor = magnitude(word)
        if factor is not None:
            amount *= factor
            continue
        named = unit(word)
        if named in TIME_UNITS:
            units.append(named)
            continue
        cardinal = NUMERALS.get(word)
        if cardinal is None:
            return None
        spelled += cardinal
    if len(units) != 1:
        return None
    if spelled:
        if numbers:
            return None  # `20 двадцять років` says two things; neither is the value
        amount = Decimal(spelled)
    if not amount:
        return None
    return f"{DURATION.lower()}:{_amount(amount)}:{units[0]}"


def _date_month(words: list[str]) -> tuple[int | None, bool]:
    """(the month named, whether the words read as a date at all).

    Every word must be either a month name or one of the fillers that says nothing about WHICH
    date it is (`року`, `р.`). A word outside both is content the grammar does not model, so the
    surface is not keyed rather than keyed on the part that happened to parse.
    """
    month: int | None = None
    for word in words:
        named = MONTHS.get(word)
        if named is not None and month is None:
            month = named
            continue
        if not stem_match(word, DATE_FILLER_STEMS):
            return None, False
    return month, True


def _date_numbers(numbers: list[Decimal]) -> tuple[int | None, int | None] | None:
    """(year, day) read off a date's literal numbers, or None when they do not read as one."""
    integers = [int(value) for value in numbers if value == value.to_integral_value()]
    if len(integers) != len(numbers):
        return None
    years = [value for value in integers if value in _YEAR_RANGE]
    days = [value for value in integers if value not in _YEAR_RANGE and 1 <= value <= _MAX_DAY]
    if len(years) > 1 or len(days) > 1 or len(years) + len(days) != len(integers):
        return None
    return (years[0] if years else None, days[0] if days else None)


def _date_parts(surface: str) -> tuple[int | None, int | None, int | None] | None:
    """(year, month, day) read off a date surface, or None when a word says it is not one."""
    folded = _fold(surface)
    iso = _ISO_DATE_RE.search(folded)
    if iso:
        return int(iso.group(1)), int(iso.group(2)), int(iso.group(3))
    dotted = _DOTTED_DATE_RE.search(folded)
    if dotted:
        return int(dotted.group(3)), int(dotted.group(2)), int(dotted.group(1))
    numbers, words = _split(folded)
    month, readable = _date_month(words)
    if not readable:
        return None
    parts = _date_numbers(numbers)
    if parts is None:
        return None
    year, day = parts
    return (year, month, day)


def _date_key(surface: str) -> str | None:
    """`2021`, `2021 року`, `1 січня 2021` and `01.01.2021` key by the point in time they name."""
    parts = _date_parts(surface)
    if parts is None:
        return None
    year, month, day = parts
    if year is None and month is None:
        return None
    if day is not None and month is None:
        return None  # a bare day names no date
    return (
        f"{DATE.lower()}:{year if year is not None else _UNKNOWN}"
        f"-{month if month is not None else _UNKNOWN}"
        f"-{day if day is not None else _UNKNOWN}"
    )


__all__ = ["VALUE_TYPES", "Lemmatizer", "default_lemmatizer", "value_key"]
