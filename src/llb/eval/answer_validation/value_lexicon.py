"""The CLOSED Ukrainian lexicon a value key is read against -- data only.

Split out of `equivalence` because it is one cohesive table family (magnitudes, units, cardinals,
month names) that a reader extends without reading the parser, and a parser that a reader follows
without scrolling past four tables.

Every entry is a decision, not a convenience. The tables exist because the pinned lemmatizer is
the wrong tool for exactly these words: it maps `рік` to `ріка` (the river), so `20 років` and
`20 рік` would part company under it. The lemmatizer still handles the open tail, where it is
right (`осіб` -> `особа`).
"""

# How far past a stem a matching form may run. `рок` + 3 admits `роки`, `років`, `роками` and
# rejects `роковини`; without the bound a stem match is a substring match wearing a prefix's name.
STEM_SLACK = 3

# Multiplier words, by stem. A magnitude is a factor on the literal number, never a unit.
MAGNITUDES: tuple[tuple[str, int], ...] = (
    ("тис", 10**3),
    ("млн", 10**6),
    ("мільйон", 10**6),
    ("млрд", 10**9),
    ("мільярд", 10**9),
    ("трлн", 10**12),
    ("трильйон", 10**12),
)

# Canonical unit -> the stems that write it. Closed on purpose (see the module docstring).
PERCENT = "%"
UNITS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (PERCENT, ("відсот", "процент")),
    ("секунда", ("секунд",)),
    ("хвилина", ("хвилин",)),
    ("година", ("годин",)),
    ("доба", ("доб", "діб")),
    ("день", ("день", "дн")),
    ("тиждень", ("тиждень", "тижн")),
    ("місяць", ("місяц", "місяч")),
    ("квартал", ("квартал",)),
    ("рік", ("рік", "рок", "роц")),
    ("десятиліття", ("десятиліт", "десятиріч")),
    ("століття", ("століт", "сторіч")),
)
# The units a DURATION may be measured in -- the rest of the table measures a QUANTITY.
TIME_UNITS: frozenset[str] = frozenset(name for name, _stems in UNITS if name != PERCENT)

# Ukrainian cardinals a duration is commonly written with, summed additively (`двадцять п'ять`).
# Nominative plus the oblique forms that actually appear before a unit; anything unlisted leaves
# the surface unparsed rather than guessed at.
NUMERALS: dict[str, int] = {
    "один": 1,
    "одна": 1,
    "одне": 1,
    "одного": 1,
    "два": 2,
    "дві": 2,
    "двох": 2,
    "три": 3,
    "трьох": 3,
    "чотири": 4,
    "чотирьох": 4,
    "п'ять": 5,
    "п'яти": 5,
    "шість": 6,
    "шести": 6,
    "сім": 7,
    "семи": 7,
    "вісім": 8,
    "восьми": 8,
    "дев'ять": 9,
    "дев'яти": 9,
    "десять": 10,
    "десяти": 10,
    "одинадцять": 11,
    "дванадцять": 12,
    "тринадцять": 13,
    "чотирнадцять": 14,
    "п'ятнадцять": 15,
    "шістнадцять": 16,
    "сімнадцять": 17,
    "вісімнадцять": 18,
    "дев'ятнадцять": 19,
    "двадцять": 20,
    "двадцяти": 20,
    "тридцять": 30,
    "тридцяти": 30,
    "сорок": 40,
    "сорока": 40,
    "п'ятдесят": 50,
    "п'ятдесяти": 50,
    "шістдесят": 60,
    "шістдесяти": 60,
    "сімдесят": 70,
    "сімдесяти": 70,
    "вісімдесят": 80,
    "вісімдесяти": 80,
    "дев'яносто": 90,
    "дев'яноста": 90,
    "сто": 100,
    "ста": 100,
}

# Month name -> ordinal, in the two forms a Ukrainian date is written in (nominative and the
# genitive `1 січня 2021`).
MONTHS: dict[str, int] = {
    "січень": 1,
    "січня": 1,
    "лютий": 2,
    "лютого": 2,
    "березень": 3,
    "березня": 3,
    "квітень": 4,
    "квітня": 4,
    "травень": 5,
    "травня": 5,
    "червень": 6,
    "червня": 6,
    "липень": 7,
    "липня": 7,
    "серпень": 8,
    "серпня": 8,
    "вересень": 9,
    "вересня": 9,
    "жовтень": 10,
    "жовтня": 10,
    "листопад": 11,
    "листопада": 11,
    "грудень": 12,
    "грудня": 12,
}
# Words a date may carry that say nothing about WHICH date it is. Anything else in a DATE surface
# is content the grammar does not model, so the surface goes unkeyed.
DATE_FILLER_STEMS: tuple[str, ...] = ("рік", "рок", "роц", "р", "числ")


def stem_match(token: str, stems: tuple[str, ...]) -> bool:
    """Whether `token` is one of these stems' forms -- a prefix, bounded by `STEM_SLACK`.

    Unbounded, a stem match is a substring match wearing a prefix's name: `рок` would take
    `роковини` as a year. The bound admits `роки`, `років`, `роками` and rejects it.
    """
    return any(token.startswith(stem) and len(token) <= len(stem) + STEM_SLACK for stem in stems)


def magnitude(token: str) -> int | None:
    """The multiplier this word names (`млн` -> 1e6), or None when it names none."""
    for stem, factor in MAGNITUDES:
        if stem_match(token, (stem,)):
            return factor
    return None


def unit(token: str) -> str | None:
    """The canonical unit this word writes, or None when the closed table does not know it."""
    for name, stems in UNITS:
        if stem_match(token, stems):
            return name
    return None


__all__ = [
    "DATE_FILLER_STEMS",
    "PERCENT",
    "MAGNITUDES",
    "MONTHS",
    "NUMERALS",
    "STEM_SLACK",
    "TIME_UNITS",
    "UNITS",
    "magnitude",
    "stem_match",
    "unit",
]
